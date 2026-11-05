# source/himloco_lab/rsl_rl/runners/amp_on_policy_runner.py
import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from ..algorithms.amp_ppo import AMPPPO
from ..modules.amp_actor_critic import AMPActorCritic
from ..env import VecEnv
from ..utils.normalizer import Normalizer
from ..algorithms.amp_discriminator import AMPDiscriminator
from ..datasets.motion_loader import AMPLoader


class AMPOnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # AMPActorCritic
        actor_critic: AMPActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.unwrapped.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg["amp_task_reward_lerp"],
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # AMPPPO
        min_std = torch.tensor(self.cfg["min_normalized_std"], device=self.device) * (
            torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])
        )
        self.alg: AMPPPO = alg_class(
            actor_critic,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        # --- 读取训练模式与 SuperMimic (World Model + BPTT) 超参数 ---
        def _get_cfg(key, default):
            return self.cfg.get(key, default) if isinstance(self.cfg, dict) else getattr(self.cfg, key, default)

        self.train_mode = _get_cfg("train_mode", "ppo")
        self.wm_window_len = _get_cfg("wm_window_len", 8)
        self.bptt_horizon = _get_cfg("bptt_horizon", 32)
        self.wm_lr = _get_cfg("wm_lr", 2e-4)
        self.wm_buffer_capacity = _get_cfg("wm_buffer_capacity", 1000)
        self.wm_batches_per_update = _get_cfg("wm_batches_per_update", 10)
        self.wm_batch_size = _get_cfg("wm_batch_size", 2048)
        self.bptt_batch_size = _get_cfg("bptt_batch_size", 4096)
        self.bptt_action_noise_std = _get_cfg("bptt_action_noise_std", 0.1)
        self.bptt_sim_steps_per_update = _get_cfg("bptt_sim_steps_per_update", 4)
        self.wm_warmup_iters = _get_cfg("wm_warmup_iters", 20)
        self.bptt_update_count = 0

        if self.train_mode == "bptt":
            from himloco_lab.rsl_rl.storage.wm_replay_buffer import WorldModelReplayBuffer

            sample_wm_state = self.env.get_wm_state()
            sample_context = self.env.get_sim_context()
            sample_clean_obs = self.env.get_clean_observations()

            wm_state_dim = sample_wm_state.shape[-1]
            context_dim = sample_context.shape[-1]
            single_obs_dim = sample_clean_obs.shape[-1]
            total_obs_frames = self.env.num_obs // single_obs_dim

            self.wm_buffer = WorldModelReplayBuffer(
                num_envs=self.env.num_envs,
                capacity_per_env=self.wm_buffer_capacity,
                state_dim=wm_state_dim,
                action_dim=self.env.num_actions,
                obs_dim=single_obs_dim,
                obs_hist_len=total_obs_frames,
                cmd_dim=3,
                context_dim=context_dim,
                device=self.device,
            )

            self.alg.init_supermimic_world_model(
                state_dim=wm_state_dim,
                action_dim=self.env.num_actions,
                context_dim=context_dim,
                wm_state_to_amp_obs_fn=self.env.wm_state_to_amp_obs,
                cmd_dim=3,
                wm_lr=self.wm_lr,
                bptt_horizon=self.bptt_horizon,
                wm_window_len=self.wm_window_len,
            )
            print(
                f"[AMPOnPolicyRunner] SuperMimic BPTT mode initialized: "
                f"wm_state_dim={wm_state_dim}, single_obs_dim={single_obs_dim}, "
                f"frames={total_obs_frames}, H={self.bptt_horizon}, k={self.wm_window_len}."
            )
        else:
            print("[AMPOnPolicyRunner] Standard Model-Free AMP-PPO mode initialized.")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        style_rewbuffer = deque(maxlen=100)
        task_rewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        task_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        style_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            if self.train_mode == "ppo":
                # ==================== 模式 A: 标准无模型 AMP-PPO ====================
                with torch.inference_mode():
                    for i in range(self.num_steps_per_env):
                        actions = self.alg.act(obs, critic_obs, amp_obs)
                        (
                            obs,
                            privileged_obs,
                            rewards,
                            dones,
                            infos,
                            termination_ids,
                            termination_privileged_obs,
                            terminal_amp_states,
                        ) = self.env.step(actions)
                        next_amp_obs = self.env.get_amp_observations()

                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones, next_amp_obs = (
                            obs.to(self.device),
                            critic_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                            next_amp_obs.to(self.device),
                        )
                        termination_ids = termination_ids.to(self.device)
                        termination_privileged_obs = termination_privileged_obs.to(self.device)
                        terminal_amp_states = terminal_amp_states.to(self.device)

                        next_amp_obs_with_term = torch.clone(next_amp_obs)
                        next_amp_obs_with_term[termination_ids] = terminal_amp_states

                        next_critic_obs = critic_obs.clone().detach()
                        next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()
                        self.task_rewards = rewards
                        rewards_all = self.alg.discriminator.predict_amp_reward(
                            amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                        )
                        rewards = rewards_all[0]
                        self.amp_rewards = rewards_all[1]
                        amp_obs = torch.clone(next_amp_obs)
                        self.alg.process_env_step(
                            rewards, dones, infos, next_critic_obs, next_amp_obs_with_term
                        )

                        if self.log_dir is not None:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            cur_reward_sum += rewards
                            task_reward_sum += self.task_rewards
                            style_reward_sum += self.amp_rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            if new_ids.numel() > 0:
                                style_rewbuffer.extend(style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                task_rewbuffer.extend(task_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                                cur_reward_sum[new_ids] = 0
                                style_reward_sum[new_ids] = 0
                                task_reward_sum[new_ids] = 0
                                cur_episode_length[new_ids] = 0

                    stop = time.time()
                    collection_time = stop - start
                    start = stop
                    self.alg.compute_returns(critic_obs)

                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_estimation_loss,
                    mean_swap_loss,
                    mean_amp_loss,
                    mean_grad_pen_loss,
                    mean_policy_pred,
                    mean_expert_pred,
                ) = self.alg.update()
                stop = time.time()
                learn_time = stop - start

            elif self.train_mode == "bptt":
                # ==================== 模式 B: SuperMimic 可微世界模型 + BPTT ====================
                mean_est_loss_accum = 0.0
                mean_swap_loss_accum = 0.0

                # 为 4096 个并行环境分配 [0.08, 0.45] 的多尺度探索噪声：
                # 低噪声环境验证当前策略，中高噪声环境打破静态接触摩擦，为世界模型持续提供加减速与抬腿样本！
                env_noise_scales = torch.linspace(
                    0.08, max(self.bptt_action_noise_std, 0.45), self.env.num_envs, device=self.device
                ).unsqueeze(-1)

                with torch.no_grad():
                    for _ in range(self.bptt_sim_steps_per_update):
                        prev_obs = obs.clone()
                        state_t = self.env.get_wm_state()
                        clean_obs_t = self.env.get_clean_observations()
                        obs_hist_t = self.env.get_obs_history_3d()
                        cmd_t = self.env.unwrapped.command_manager.get_command("base_velocity")[:, :3]
                        xi_t = self.env.get_sim_context()

                        det_actions = self.alg.actor_critic.act_inference(obs)
                        noisy_actions = det_actions + torch.randn_like(det_actions) * env_noise_scales

                        (
                            obs,
                            privileged_obs,
                            rewards,
                            dones,
                            infos,
                            termination_ids,
                            termination_privileged_obs,
                            terminal_amp_states,
                        ) = self.env.step(noisy_actions)

                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones = (
                            obs.to(self.device),
                            critic_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        termination_ids = termination_ids.to(self.device)
                        termination_privileged_obs = termination_privileged_obs.to(self.device)
                        terminal_amp_states = terminal_amp_states.to(self.device)

                        self.env.update_clean_obs_history(termination_ids)
                        next_state_t = self.env.get_wm_state()
                        next_clean_obs_t = self.env.get_clean_observations()

                        # 还原终止环境在自动 Reset 之前的真实 46 维物理状态与 42 维 AMP 状态
                        next_state_t_fixed = self.env.apply_terminal_amp_to_wm_state(
                            next_state_t, termination_ids, terminal_amp_states
                        )
                        amp_s = self.env.wm_state_to_amp_obs(state_t)
                        amp_next_s = self.env.wm_state_to_amp_obs(next_state_t_fixed)

                        next_critic_obs = critic_obs.clone().detach()
                        next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                        resets_mask = dones > 0
                        if "time_outs" in infos:
                            time_outs_mask = infos["time_outs"].to(self.device).bool()
                            falls_mask = resets_mask & (~time_outs_mask)
                        else:
                            falls_mask = resets_mask

                        self.wm_buffer.insert(
                            state_t,
                            next_state_t_fixed,
                            noisy_actions,
                            clean_obs_t,
                            next_clean_obs_t,
                            obs_hist_t,
                            rewards,
                            falls_mask,
                            resets_mask,
                            cmd_t,
                            xi_t,
                        )
                        if getattr(self.alg, "amp_storage", None) is not None:
                            self.alg.amp_storage.insert(amp_s, amp_next_s)

                        # 使用 1e-3 学习率更新 HIMEstimator，确保速度估计误差快速收敛
                        if hasattr(self.alg.actor_critic, "estimator"):
                            with torch.enable_grad():
                                est_loss, swap_loss = self.alg.actor_critic.estimator.update(
                                    prev_obs, next_critic_obs, lr=1e-3
                                )
                                mean_est_loss_accum += est_loss / self.bptt_sim_steps_per_update
                                mean_swap_loss_accum += swap_loss / self.bptt_sim_steps_per_update

                        self.task_rewards = rewards
                        rewards_all = self.alg.discriminator.predict_amp_reward(
                            amp_s, amp_next_s, rewards, normalizer=self.alg.amp_normalizer
                        )
                        step_total_rewards = rewards_all[0]
                        self.amp_rewards = rewards_all[1]

                        if self.log_dir is not None:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            cur_reward_sum += step_total_rewards
                            task_reward_sum += self.task_rewards
                            style_reward_sum += self.amp_rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            if new_ids.numel() > 0:
                                style_rewbuffer.extend(style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                task_rewbuffer.extend(task_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                                cur_reward_sum[new_ids] = 0
                                style_reward_sum[new_ids] = 0
                                task_reward_sum[new_ids] = 0
                                cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                if self.wm_buffer.size > self.wm_window_len + 2:
                    # 阶段 (2): 拟合可微世界模型与 AMP 判别器
                    wm_metrics = self.alg.update_world_model(
                        self.wm_buffer,
                        num_batches=self.wm_batches_per_update,
                        batch_size=self.wm_batch_size,
                    )
                    self.bptt_update_count += 1

                    # 阶段 (3): 世界模型完成预热 (wm_warmup_iters=20) 后，开启 H=32 步 BPTT 策略更新
                    if self.bptt_update_count > self.wm_warmup_iters:
                        bptt_metrics = self.alg.update_bptt(
                            self.wm_buffer,
                            batch_size=self.bptt_batch_size,
                        )
                        mean_value_loss = bptt_metrics["bptt_critic_loss"]
                        mean_surrogate_loss = -bptt_metrics["bptt_actor_return"]
                        mean_policy_pred = bptt_metrics["bptt_mean_survival_H"]
                    else:
                        mean_value_loss = 0.0
                        mean_surrogate_loss = 0.0
                        mean_policy_pred = 1.0

                    mean_estimation_loss = float(mean_est_loss_accum)
                    mean_swap_loss = float(mean_swap_loss_accum)
                    mean_amp_loss = wm_metrics["wm_loss_total"]
                    mean_grad_pen_loss = wm_metrics["wm_loss_state"]
                    mean_expert_pred = wm_metrics["wm_loss_reward"]
                else:
                    mean_value_loss = 0.0
                    mean_surrogate_loss = 0.0
                    mean_estimation_loss = float(mean_est_loss_accum)
                    mean_swap_loss = float(mean_swap_loss_accum)
                    mean_amp_loss = 0.0
                    mean_grad_pen_loss = 0.0
                    mean_policy_pred = 1.0
                    mean_expert_pred = 0.0

                stop = time.time()
                learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        steps_per_iter = (
            self.bptt_sim_steps_per_update if self.train_mode == "bptt" else self.num_steps_per_env
        )
        self.tot_timesteps += steps_per_iter * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            steps_per_iter * self.env.num_envs / max(locs["collection_time"] + locs["learn_time"], 1e-6)
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/Estimation Loss", locs["mean_estimation_loss"], locs["it"])
        self.writer.add_scalar("Loss/Swap Loss", locs["mean_swap_loss"], locs["it"])
        self.writer.add_scalar("Loss/AMP", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/AMP_grad", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # 防御性安全均值函数：绝不因任何 buffer 为空而抛出 StatisticsError
        def safe_mean(buf):
            return statistics.mean(buf) if (buf is not None and len(buf) > 0) else 0.0

        has_data = len(locs["rewbuffer"]) > 0
        mean_rew = safe_mean(locs["rewbuffer"])
        mean_task_rew = safe_mean(locs["task_rewbuffer"])
        mean_style_rew = safe_mean(locs["style_rewbuffer"])
        mean_len = safe_mean(locs["lenbuffer"])

        if has_data:
            self.writer.add_scalar("Train/mean_style_reward", mean_style_rew, locs["it"])
            self.writer.add_scalar("Train/mean_task_reward", mean_task_rew, locs["it"])
            self.writer.add_scalar("Train/mean_reward", mean_rew, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", mean_len, locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", mean_rew, self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", mean_len, self.tot_time)

        str_header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} ({self.train_mode.upper()}) \033[0m "

        if has_data:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                f"""{'AMP / WM loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'AMP grad / WM state loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'AMP policy pred / Survival_H:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'AMP expert pred / WM rew loss:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {mean_rew:.2f}\n"""
                f"""{'Mean task reward:':>{pad}} {mean_task_rew:.2f}\n"""
                f"""{'Mean style reward:':>{pad}} {mean_style_rew:.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {mean_len:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                f"""{'AMP / WM loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'AMP grad / WM state loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        state = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if hasattr(self.alg, "world_model"):
            state["world_model_state_dict"] = self.alg.world_model.state_dict()
        torch.save(state, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if hasattr(self.alg, "world_model") and "world_model_state_dict" in loaded_dict:
            self.alg.world_model.load_state_dict(loaded_dict["world_model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(
                loaded_dict["estimator_optimizer_state_dict"]
            )
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
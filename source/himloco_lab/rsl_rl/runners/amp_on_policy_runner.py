# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import deque
import inspect
import math
import os
import statistics
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from himloco_lab.rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from himloco_lab.rsl_rl.algorithms.amp_ppo import AMPPPO
from himloco_lab.rsl_rl.datasets.motion_loader import AMPLoader
from himloco_lab.rsl_rl.env import VecEnv
from himloco_lab.rsl_rl.modules import AMPActorCritic, HIMActorCritic
from himloco_lab.rsl_rl.utils.normalizer import Normalizer


def _cosine_lr(base_lr: float, step: int, decay_steps: int = 250000) -> float:
    """Cosine learning rate schedule matching SuperMimic (supermimic/core/batching.py)."""
    t = min(max(step, 0), max(decay_steps, 1)) / max(decay_steps, 1)
    return base_lr * (0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * t)))


def _exploration_std(step: int, initial: float, final: float, decay_steps: int) -> float:
    """Linearly decaying exploration noise std matching SuperMimic (supermimic/core/bootstrap.py)."""
    fraction = min(max(step, 0) / max(decay_steps, 1), 1.0)
    return initial + fraction * (final - initial)


class AMPOnPolicyRunner:
    """Runner supporting both standard AMP-PPO ('ppo') and SuperMimic WMRL ('bptt') training modes."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.cfg = dict(train_cfg)
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None

        # 解析训练模式: 优先读取 train_mode ("ppo" / "bptt")，其次兼容 use_bptt 布尔值
        raw_mode = self.alg_cfg.get("train_mode", self.cfg.get("train_mode", None))
        if raw_mode is not None:
            self.train_mode = str(raw_mode).strip().lower()
        else:
            use_bptt = bool(self.alg_cfg.get("use_bptt", self.cfg.get("use_bptt", False)))
            self.train_mode = "bptt" if use_bptt else "ppo"

        if self.train_mode not in ("ppo", "bptt"):
            raise ValueError(f"Unsupported train_mode '{self.train_mode}'. Expected 'ppo' or 'bptt'.")

        # 解析 Actor / Critic 观测维度
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        num_one_step_obs = getattr(self.env, "num_one_step_obs", 45)
        actor_critic_class_name = self.policy_cfg.pop("class_name", "AMPActorCritic")
        actor_critic_class = eval(actor_critic_class_name)
        actor_critic: AMPActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # 初始化 AMP 专家动作加载器、特征归一化器与判别器
        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.dt,
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

        # 计算最小动作噪声标准差约束
        min_normalized_std = self.cfg.get("min_normalized_std", [0.05] * self.env.num_actions)
        if hasattr(self.env, "dof_pos_limits"):
            dof_range = torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]).to(self.device)
            min_std = torch.tensor(min_normalized_std, device=self.device, dtype=torch.float32) * dof_range
        else:
            min_std = torch.tensor(min_normalized_std, device=self.device, dtype=torch.float32)

        # 实例化算法类 (自动过滤签名，兼容不同版本的 AMPPPO.__init__)
        alg_class_name = self.alg_cfg.pop("class_name", "AMPPPO")
        alg_class = eval(alg_class_name)
        alg_sig = inspect.signature(alg_class.__init__)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in alg_sig.parameters.values()
        )
        if has_var_keyword:
            filtered_alg_cfg = dict(self.alg_cfg)
        else:
            valid_keys = set(alg_sig.parameters.keys()) - {
                "self",
                "actor_critic",
                "discriminator",
                "amp_data",
                "amp_normalizer",
                "min_std",
                "device",
            }
            filtered_alg_cfg = {k: v for k, v in self.alg_cfg.items() if k in valid_keys}

        self.alg: AMPPPO = alg_class(
            actor_critic=actor_critic,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            min_std=min_std,
            device=self.device,
            **filtered_alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # 初始化 PPO Rollout Storage (自动适配 5 参数或 6 参数签名)
        storage_params = list(inspect.signature(self.alg.init_storage).parameters.values())
        required_pos_args = [
            p
            for p in storage_params
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required_pos_args) <= 5:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_obs],
                [self.env.num_privileged_obs],
                [self.env.num_actions],
            )
        else:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_obs],
                [self.env.num_privileged_obs],
                [self.env.num_actions],
                [amp_data.observation_dim],
            )

        # 若当前为 BPTT 模式，初始化 SuperMimic 世界模型、BPTTValueModel 与 WMReplayBuffer
        if self.train_mode == "bptt":
            self._init_bptt_components()

        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

    # ==========================================================================
    # 内部辅助方法：初始化 BPTT 组件与环境状态读取
    # ==========================================================================
    def _init_bptt_components(self):
        unwrapped = self.env.unwrapped
        robot = unwrapped.scene["robot"]
        default_q = robot.data.default_joint_pos[0].clone().to(self.device)
        joint_limits = robot.data.soft_joint_pos_limits[0].clone().to(self.device)
        nominal_height = float(
            self.alg_cfg.get(
                "nominal_height",
                getattr(unwrapped, "nominal_base_height", robot.data.default_root_state[0, 2].item()),
            )
        )
        min_height = float(self.alg_cfg.get("min_height", max(0.25, nominal_height * 0.45)))
        max_tilt_deg = float(self.alg_cfg.get("max_tilt_deg", 65.0))
        num_one_step_obs = getattr(self.env, "num_one_step_obs", 45)

        init_sig = inspect.signature(self.alg.init_supermimic_bptt)
        init_kwargs = dict(
            num_envs=self.env.num_envs,
            num_joints=self.env.num_actions,
            num_one_step_obs=num_one_step_obs,
            default_joint_pos=default_q,
            joint_limits=joint_limits,
            wm_width=int(self.alg_cfg.get("wm_width", 2048)),
            wm_lr=float(self.alg_cfg.get("wm_lr", 2e-4)),
            actor_lr=float(self.alg_cfg.get("actor_lr", self.alg_cfg.get("learning_rate", 2e-4))),
            critic_lr=float(self.alg_cfg.get("critic_lr", 2e-4)),
            horizon=int(self.alg_cfg.get("bptt_horizon", 32)),
            wm_buffer_steps=int(self.alg_cfg.get("wm_buffer_steps", 256)),
        )
        if "nominal_height" in init_sig.parameters:
            init_kwargs["nominal_height"] = nominal_height
        if "min_height" in init_sig.parameters:
            init_kwargs["min_height"] = min_height
        if "max_tilt_deg" in init_sig.parameters:
            init_kwargs["max_tilt_deg"] = max_tilt_deg
        if "dt" in init_sig.parameters:
            init_kwargs["dt"] = float(self.env.dt)

        self.alg.init_supermimic_bptt(**init_kwargs)

        if hasattr(self.alg, "world_model") and self.alg.world_model is not None:
            self.alg.world_model.nominal_height = nominal_height
            self.alg.world_model.min_height = min_height
            self.alg.world_model.cos_max_tilt = math.cos(math.radians(max_tilt_deg))
            self.alg.world_model.dt = float(self.env.dt)

    def _get_obs_tensor(self) -> torch.Tensor:
        obs_out = self.env.get_observations()
        if isinstance(obs_out, tuple):
            return obs_out[0].to(self.device)
        return obs_out.to(self.device)

    def _get_physics_state(self) -> torch.Tensor:
        if hasattr(self.env, "get_physics_state"):
            return self.env.get_physics_state().to(self.device)
        if hasattr(self.env.unwrapped, "read_physics_state"):
            return self.env.unwrapped.read_physics_state().to(self.device)
        robot = self.env.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w - self.env.unwrapped.scene.env_origins
        return torch.cat(
            (
                root_pos,
                robot.data.root_quat_w,
                robot.data.root_lin_vel_w,
                robot.data.root_ang_vel_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
            ),
            dim=-1,
        ).to(self.device)

    def _get_post_step_physics_state(self, infos: dict) -> torch.Tensor:
        if isinstance(infos, dict) and "post_step_physics_state" in infos:
            return infos["post_step_physics_state"].to(self.device)
        if hasattr(self.env, "get_post_step_physics_state"):
            return self.env.get_post_step_physics_state().to(self.device)
        if hasattr(self.env.unwrapped, "get_post_step_physics_state"):
            return self.env.unwrapped.get_post_step_physics_state().to(self.device)
        return self._get_physics_state()

    def _get_velocity_commands(self) -> torch.Tensor:
        if hasattr(self.env, "get_velocity_commands"):
            return self.env.get_velocity_commands().to(self.device)
        if hasattr(self.env.unwrapped, "get_velocity_commands"):
            return self.env.unwrapped.get_velocity_commands().to(self.device)
        return self.env.unwrapped.command_manager.get_command("base_velocity").clone().to(self.device)

    def _get_feet_contact(self, infos: dict) -> torch.Tensor:
        if isinstance(infos, dict) and "feet_contact" in infos:
            return infos["feet_contact"].to(self.device)
        if hasattr(self.env, "get_feet_contact"):
            return self.env.get_feet_contact().to(self.device)
        if hasattr(self.env.unwrapped, "read_feet_contact"):
            return self.env.unwrapped.read_feet_contact().to(self.device)
        return torch.zeros((self.env.num_envs, 4), device=self.device, dtype=torch.float32)

    def _unpack_env_step(self, step_out: tuple):
        """Robustly unpacks environment step outputs of any length (4, 5, 7, 8, etc.)."""
        if not isinstance(step_out, (tuple, list)):
            return step_out, None, None, None, {}, torch.tensor([], dtype=torch.long, device=self.device), None

        length = len(step_out)
        if length == 8:
            obs, privileged_obs, rewards, dones, infos, _, reset_env_ids, terminal_amp_states = step_out
        elif length == 7:
            obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = step_out
        elif length == 5:
            obs, privileged_obs, rewards, dones, infos = step_out
            reset_env_ids = dones.nonzero(as_tuple=False).flatten() if isinstance(dones, torch.Tensor) else torch.tensor([], dtype=torch.long, device=self.device)
            terminal_amp_states = None
        elif length == 4:
            obs, rewards, dones, infos = step_out
            privileged_obs = infos.get("observations", {}).get("critic", None) if isinstance(infos, dict) else None
            reset_env_ids = dones.nonzero(as_tuple=False).flatten() if isinstance(dones, torch.Tensor) else torch.tensor([], dtype=torch.long, device=self.device)
            terminal_amp_states = None
        else:
            obs = step_out[0]
            privileged_obs = step_out[1] if length > 1 else None
            rewards = step_out[2] if length > 2 else None
            dones = step_out[3] if length > 3 else None
            infos = step_out[4] if length > 4 else {}
            reset_env_ids = dones.nonzero(as_tuple=False).flatten() if isinstance(dones, torch.Tensor) else torch.tensor([], dtype=torch.long, device=self.device)
            terminal_amp_states = step_out[7] if length > 7 else (step_out[6] if length > 6 else None)

        # 强制将 reset_env_ids 转换为标准 PyTorch LongTensor，防止 Tensor 索引报错
        if reset_env_ids is not None:
            if not isinstance(reset_env_ids, torch.Tensor):
                reset_env_ids = torch.tensor(reset_env_ids, dtype=torch.long, device=self.device)
            else:
                reset_env_ids = reset_env_ids.to(device=self.device, dtype=torch.long)

        return obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states

    # ==========================================================================
    # 训练统一入口 (通过 train_mode 分发至 learn_ppo 或 learn_bptt)
    # ==========================================================================
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        print(f"[AMPOnPolicyRunner] Starting training in '{self.train_mode.upper()}' mode...", flush=True)
        if self.train_mode == "bptt":
            self.learn_bptt(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
        else:
            self.learn_ppo(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

    # ==========================================================================
    # 模式 A: 标准 AMP-PPO 训练模式
    # ==========================================================================
    def learn_ppo(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self._get_obs_tensor()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs
        amp_obs = amp_obs.to(self.device)

        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        proc_params = list(inspect.signature(self.alg.process_env_step).parameters.keys())

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # 1. 环境交互收集 Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    step_out = self.env.step(actions)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = (
                        self._unpack_env_step(step_out)
                    )
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    next_amp_obs = self.env.get_amp_observations().to(self.device)

                    # 还原终止环境的 AMP 观测
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    if terminal_amp_states is not None and len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states.to(self.device)

                    # 融合任务奖励与 AMP 判别器风格奖励
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )[0]
                    amp_obs = torch.clone(next_amp_obs)

                    if "next_critic_obs" in proc_params and len(proc_params) >= 5:
                        if proc_params.index("next_critic_obs") == 3:
                            self.alg.process_env_step(rewards, dones, infos, critic_obs, next_amp_obs_with_term)
                        else:
                            self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term, critic_obs)
                    else:
                        self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        if isinstance(infos, dict):
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # 2. 计算 GAE Returns
                start = stop
                self.alg.compute_returns(critic_obs)

            # 3. 执行 PPO 与 AMP 判别器更新
            update_res = self.alg.update()
            if len(update_res) == 8:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_estimation_loss,
                    mean_swap_loss,
                    mean_amp_loss,
                    mean_grad_pen_loss,
                    mean_policy_pred,
                    mean_expert_pred,
                ) = update_res
            else:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_amp_loss,
                    mean_grad_pen_loss,
                    mean_policy_pred,
                    mean_expert_pred,
                ) = update_res[:6]
                mean_estimation_loss, mean_swap_loss = 0.0, 0.0

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            self.current_learning_iteration = it + 1

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ==========================================================================
    # 模式 B: SuperMimic 世界模型 + BPTT 训练模式
    # ==========================================================================
    def learn_bptt(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        if not hasattr(self.alg, "world_model") or self.alg.world_model is None:
            self._init_bptt_components()

        # 读取 SuperMimic BPTT 超参数
        warmup_steps = int(self.alg_cfg.get("wm_warmup_steps", 64))
        pretrain_updates = int(self.alg_cfg.get("wm_pretrain_updates", 500))
        collect_steps = int(self.alg_cfg.get("bptt_collect_steps", 3))
        bptt_batch_size = int(self.alg_cfg.get("bptt_batch_size", 1024))
        wm_batch_size = int(self.alg_cfg.get("wm_batch_size", 2048))
        wm_batches = int(self.alg_cfg.get("wm_batches", 10))
        critic_epochs = int(self.alg_cfg.get("num_learning_epochs", 5))
        critic_minibatches = int(self.alg_cfg.get("num_mini_batches", 4))
        exp_initial = float(self.alg_cfg.get("exploration_initial", self.alg_cfg.get("exploration_std", 0.1)))
        exp_final = float(self.alg_cfg.get("exploration_final", exp_initial))
        exp_decay = int(self.alg_cfg.get("exploration_decay_updates", 10000))
        lr_decay_updates = int(self.alg_cfg.get("lr_decay_updates", 250000))
        base_actor_lr = float(self.alg_cfg.get("actor_lr", self.alg_cfg.get("learning_rate", 2e-4)))
        base_wm_lr = float(self.alg_cfg.get("wm_lr", 2e-4))
        base_critic_lr = float(self.alg_cfg.get("critic_lr", 2e-4))

        obs_history = self._get_obs_tensor()
        amp_obs = self.env.get_amp_observations().to(self.device)
        prev_action = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)
        pre_pre_action = torch.zeros_like(prev_action)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        def collect_env_steps(n_steps: int, noise_std: float):
            nonlocal obs_history, amp_obs, prev_action, pre_pre_action, cur_reward_sum, cur_episode_length
            self.alg.actor_critic.eval()
            with torch.no_grad():
                for _ in range(n_steps):
                    state = self._get_physics_state()
                    cmd = self._get_velocity_commands()

                    det_action = self.alg.actor_critic.act_inference(obs_history)
                    action = det_action + noise_std * torch.randn_like(det_action)

                    step_out = self.env.step(action)
                    next_obs, _, task_rewards, dones, infos, reset_env_ids, terminal_amp_states = (
                        self._unpack_env_step(step_out)
                    )
                    if isinstance(infos, dict) and "observations" in infos and "obs_history" in infos["observations"]:
                        next_obs_history = infos["observations"]["obs_history"].to(self.device)
                    else:
                        next_obs_history = next_obs.to(self.device)

                    task_rewards = task_rewards.to(self.device).view(-1)
                    dones = dones.to(self.device).bool().view(-1)

                    # 还原 Reset 环境的终端 AMP 状态并写入 AMP 经验池
                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    if terminal_amp_states is not None and len(reset_env_ids) > 0:
                        # 确保 reset_env_ids 是合法的 torch.LongTensor
                        if not isinstance(reset_env_ids, torch.Tensor):
                            reset_env_ids = torch.tensor(reset_env_ids, dtype=torch.long, device=self.device)
                        else:
                            reset_env_ids = reset_env_ids.to(device=self.device, dtype=torch.long)

                        term_states = terminal_amp_states.to(self.device)
                        # 如果 next_amp_obs_with_term 带有时间/历史维度 [num_envs, history_len, feat_dim]，
                        # 而 term_states 是单帧 [num_reset_envs, feat_dim]，则将其填入最新帧或整段历史
                        if next_amp_obs_with_term.ndim == 3 and term_states.ndim == 2:
                            # 填入最后一帧（最新帧）
                            next_amp_obs_with_term[reset_env_ids, -1, :] = term_states
                        elif next_amp_obs_with_term.ndim == 3 and term_states.ndim == 3:
                            next_amp_obs_with_term[reset_env_ids] = term_states
                        else:
                            next_amp_obs_with_term[reset_env_ids] = term_states
                        # next_amp_obs_with_term[reset_env_ids] = terminal_amp_states.to(self.device)

                    self.alg.amp_storage.insert(amp_obs, next_amp_obs_with_term)

                    # 计算融合了 AMP 风格奖励与环境任务奖励的综合奖励
                    combined_rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, task_rewards, normalizer=self.alg.amp_normalizer
                    )[0].view(-1)

                    time_outs = (
                        infos.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().view(-1)
                        if isinstance(infos, dict)
                        else torch.zeros_like(dones)
                    )
                    terminated = dones & (~time_outs)
                    truncated = time_outs

                    next_state = self._get_post_step_physics_state(infos)
                    feet_contact = self._get_feet_contact(infos)

                    # 仅使用未发生重置的有效步更新世界模型的 RunningNorm 统计量
                    valid_mask = ~(terminated | truncated)
                    self.alg.world_model.update_stats(
                        state,
                        next_state,
                        action,
                        reward=combined_rewards,
                        valid_mask=valid_mask,
                    )

                    self.alg.wm_replay.add(
                        state=state,
                        next_state=next_state,
                        action=action,
                        prev_action=prev_action,
                        pre_pre_action=pre_pre_action,
                        obs_history=obs_history,
                        next_obs_history=next_obs_history,
                        command=cmd,
                        reward=combined_rewards,
                        terminated=terminated,
                        truncated=truncated,
                        feet_contact=feet_contact,
                    )

                    if self.log_dir is not None:
                        if isinstance(infos, dict):
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                        cur_reward_sum += combined_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                    pre_pre_action = torch.where(dones[:, None], torch.zeros_like(pre_pre_action), prev_action)
                    prev_action = torch.where(dones[:, None], torch.zeros_like(prev_action), action)
                    obs_history = next_obs_history
                    amp_obs = torch.clone(next_amp_obs)

        # 阶段 1: 收集初始 Warmup 步以填充 WMReplayBuffer 和统计量
        if self.alg.wm_replay.size < warmup_steps:
            need_steps = max(warmup_steps - self.alg.wm_replay.size, 8)
            print(f"[SuperMimic-BPTT] Collecting {need_steps} warmup steps...", flush=True)
            collect_env_steps(need_steps, exp_initial)
            ep_infos.clear()

        # 阶段 2: 预训练世界模型与 AMP 判别器 500 次迭代 (冻结 Actor)
        if self.current_learning_iteration == 0 and pretrain_updates > 0:
            print(
                f"[SuperMimic-BPTT] Pretraining World Model & AMP Discriminator for {pretrain_updates} iterations...",
                flush=True,
            )
            for pre_it in range(1, pretrain_updates + 1):
                collect_env_steps(collect_steps, exp_initial)
                wm_info = self.alg.update_world_model(wm_batches=wm_batches, wm_batch_size=wm_batch_size)
                disc_info = self._step_amp_discriminator(batch_size=min(bptt_batch_size, 512))
                if pre_it == 1 or pre_it % 50 == 0 or pre_it == pretrain_updates:
                    print(
                        f"  [Pretrain {pre_it:4d}/{pretrain_updates}] "
                        f"WM Loss: {wm_info['wm_loss']:.5f} (Phys: {wm_info.get('wm_physics_loss', 0.0):.5f}, "
                        f"Aux: {wm_info.get('wm_aux_loss', 0.0):.5f}) | "
                        f"AMP Loss: {disc_info['amp_loss']:.4f}",
                        flush=True,
                    )
                if self.writer is not None and pre_it % 10 == 0:
                    self.writer.add_scalar("Bootstrap/wm_loss", wm_info["wm_loss"], pre_it)
                    self.writer.add_scalar("Bootstrap/amp_loss", disc_info["amp_loss"], pre_it)
            ep_infos.clear()

        # 阶段 3: 正式联合训练 (收集真实交互 -> 更新 WM & AMP -> 32步 BPTT 更新 Actor -> GAE 更新 Critic)
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # 余弦学习率调度与探索噪声衰减
            actor_lr = _cosine_lr(base_actor_lr, it, lr_decay_updates)
            wm_lr = _cosine_lr(base_wm_lr, it, lr_decay_updates)
            critic_lr = _cosine_lr(base_critic_lr, it, lr_decay_updates)
            for group in self.alg.bptt_actor_optimizer.param_groups:
                group["lr"] = actor_lr
            for group in self.alg.wm_optimizer.param_groups:
                group["lr"] = wm_lr
            for group in self.alg.bptt_critic_optimizer.param_groups:
                group["lr"] = critic_lr

            current_noise_std = _exploration_std(it, exp_initial, exp_final, exp_decay)

            # 1. 收集真实环境步
            collect_env_steps(collect_steps, current_noise_std)
            stop = time.time()
            collection_time = stop - start

            # 2. 更新 AMP 判别器与执行 BPTT 联合更新
            start = stop
            disc_info = self._step_amp_discriminator(batch_size=min(bptt_batch_size, 512))
            metrics = self.alg.update_bptt_step(
                batch_size=bptt_batch_size,
                wm_batches=wm_batches,
                wm_batch_size=wm_batch_size,
                critic_epochs=critic_epochs,
                critic_minibatches=critic_minibatches,
            )
            metrics.setdefault("amp_loss", disc_info["amp_loss"])
            metrics.setdefault("amp_grad_pen", disc_info["amp_grad_pen"])

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log_bptt(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            self.current_learning_iteration = it + 1

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _step_amp_discriminator(self, batch_size: int = 512) -> dict[str, float]:
        """在 BPTT 模式下更新 AMP 判别器与特征 Normalizer。"""
        if hasattr(self.alg, "update_amp_discriminator"):
            return self.alg.update_amp_discriminator(batch_size=batch_size)

        if self.alg.amp_storage.num_samples < batch_size:
            return {"amp_loss": 0.0, "amp_grad_pen": 0.0}

        if not hasattr(self, "_bptt_disc_optimizer"):
            disc_lr = float(self.alg_cfg.get("discriminator_lr", 1e-5))
            self._bptt_disc_optimizer = torch.optim.Adam(
                [
                    {"params": self.alg.discriminator.trunk.parameters(), "weight_decay": 1e-3},
                    {"params": self.alg.discriminator.amp_linear.parameters(), "weight_decay": 1e-1},
                ],
                lr=disc_lr,
            )

        policy_gen = self.alg.amp_storage.feed_forward_generator(1, batch_size)
        expert_gen = self.alg.amp_data.feed_forward_generator(1, batch_size)
        policy_state, policy_next_state = next(policy_gen)
        expert_state, expert_next_state = next(expert_gen)

        policy_state_raw = policy_state.clone()
        expert_state_raw = expert_state.clone()

        if self.alg.amp_normalizer is not None:
            with torch.no_grad():
                policy_state = self.alg.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.alg.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.alg.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.alg.amp_normalizer.normalize_torch(expert_next_state, self.device)

        self.alg.discriminator.train()
        policy_d = self.alg.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.alg.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        amp_loss = 0.5 * ((expert_d - 1.0).square().mean() + (policy_d + 1.0).square().mean())
        grad_pen = self.alg.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

        self._bptt_disc_optimizer.zero_grad(set_to_none=True)
        (amp_loss + grad_pen).backward()
        torch.nn.utils.clip_grad_norm_(self.alg.discriminator.parameters(), 1.0)
        self._bptt_disc_optimizer.step()

        if self.alg.amp_normalizer is not None:
            self.alg.amp_normalizer.update(policy_state_raw.detach().cpu().numpy())
            self.alg.amp_normalizer.update(expert_state_raw.detach().cpu().numpy())

        return {"amp_loss": float(amp_loss.detach()), "amp_grad_pen": float(grad_pen.detach())}

    # ==========================================================================
    # 终端与 TensorBoard 日志输出
    # ==========================================================================
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / max(locs["collection_time"] + locs["learn_time"], 1e-6))

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
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        tot_iter = locs.get("tot_iter", self.current_learning_iteration + locs.get("num_learning_iterations", 1))
        str_header = f" \033[1m Learning iteration {locs['it']}/{tot_iter} (PPO) \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string, flush=True)

    def log_bptt(self, locs: dict, width: int = 80, pad: int = 35):
        collect_steps = locs["collect_steps"]
        self.tot_timesteps += collect_steps * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        metrics = locs["metrics"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(collect_steps * self.env.num_envs / max(iteration_time, 1e-6))

        self.writer.add_scalar("Loss/world_model_total", metrics.get("wm_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/world_model_physics", metrics.get("wm_physics_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/world_model_contact", metrics.get("wm_contact_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/world_model_aux", metrics.get("wm_aux_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/bptt_critic", metrics.get("bptt_critic_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/Estimation Loss", metrics.get("estimator_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/AMP", metrics.get("amp_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/AMP_grad", metrics.get("amp_grad_pen", 0.0), locs["it"])
        self.writer.add_scalar("BPTT/imagined_return", metrics.get("bptt_imagined_return", 0.0), locs["it"])
        self.writer.add_scalar("BPTT/survival_rate", metrics.get("bptt_survival_rate", 0.0), locs["it"])
        self.writer.add_scalar("BPTT/actor_grad_norm", metrics.get("bptt_actor_grad_norm", 0.0), locs["it"])
        self.writer.add_scalar("Policy/exploration_std", locs["current_noise_std"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", locs["actor_lr"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        tot_iter = locs.get("tot_iter", self.current_learning_iteration + locs.get("num_learning_iterations", 1))
        str_header = f" \033[1m Learning iteration {locs['it']}/{tot_iter} (SuperMimic-BPTT) \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{str_header.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collect: {locs['collection_time']:.3f}s, learn: {locs['learn_time']:.3f}s)\n"""
            f"""{'World Model loss (8-step):':>{pad}} {metrics.get('wm_loss', 0.0):.4f} (phys: {metrics.get('wm_physics_loss', 0.0):.4f}, aux: {metrics.get('wm_aux_loss', 0.0):.4f})\n"""
            f"""{'BPTT Imagined Return (H=32):':>{pad}} {metrics.get('bptt_imagined_return', 0.0):.4f}\n"""
            f"""{'BPTT Survival Rate (H=32):':>{pad}} {100.0 * metrics.get('bptt_survival_rate', 0.0):.2f}%\n"""
            f"""{'BPTT Critic GAE loss:':>{pad}} {metrics.get('bptt_critic_loss', 0.0):.4f}\n"""
            f"""{'Actor Grad Norm:':>{pad}} {metrics.get('bptt_actor_grad_norm', 0.0):.4f}\n"""
            f"""{'Estimator loss:':>{pad}} {metrics.get('estimator_loss', 0.0):.4f}\n"""
            f"""{'AMP Discriminator loss:':>{pad}} {metrics.get('amp_loss', 0.0):.4f}\n"""
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{'Mean real episode reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean real episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total real timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
        )
        print(log_string, flush=True)

    def save(self, path: str, infos: dict | None = None):
        state = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
            "train_mode": self.train_mode,
        }
        if hasattr(self.alg, "world_model") and self.alg.world_model is not None:
            state.update(
                {
                    "world_model_state_dict": self.alg.world_model.state_dict(),
                    "bptt_critic_state_dict": self.alg.bptt_critic.state_dict(),
                    "wm_optimizer_state_dict": self.alg.wm_optimizer.state_dict(),
                    "bptt_actor_optimizer_state_dict": self.alg.bptt_actor_optimizer.state_dict(),
                    "bptt_critic_optimizer_state_dict": self.alg.bptt_critic_optimizer.state_dict(),
                }
            )
        torch.save(state, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if "discriminator_state_dict" in loaded_dict and self.alg.discriminator is not None:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if "amp_normalizer" in loaded_dict and loaded_dict["amp_normalizer"] is not None:
            self.alg.amp_normalizer = loaded_dict["amp_normalizer"]

        if hasattr(self.alg, "world_model") and self.alg.world_model is not None:
            if "world_model_state_dict" in loaded_dict:
                self.alg.world_model.load_state_dict(loaded_dict["world_model_state_dict"])
            if "bptt_critic_state_dict" in loaded_dict:
                self.alg.bptt_critic.load_state_dict(loaded_dict["bptt_critic_state_dict"])

        if load_optimizer:
            if "optimizer_state_dict" in loaded_dict:
                try:
                    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                except Exception:
                    pass
            if hasattr(self.alg, "world_model") and self.alg.world_model is not None:
                if "wm_optimizer_state_dict" in loaded_dict:
                    self.alg.wm_optimizer.load_state_dict(loaded_dict["wm_optimizer_state_dict"])
                if "bptt_actor_optimizer_state_dict" in loaded_dict:
                    self.alg.bptt_actor_optimizer.load_state_dict(loaded_dict["bptt_actor_optimizer_state_dict"])
                if "bptt_critic_optimizer_state_dict" in loaded_dict:
                    self.alg.bptt_critic_optimizer.load_state_dict(loaded_dict["bptt_critic_optimizer_state_dict"])

        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device: str | None = None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
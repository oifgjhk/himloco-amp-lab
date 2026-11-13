# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from himloco_lab.rsl_rl.modules import AMPActorCritic
from himloco_lab.rsl_rl.modules.world_model import (
    BPTTValueModel,
    WorldModel,
    compute_wm_rollout_loss,
    critic_input,
    freeze_parameters,
    mv,
    quat_to_mat,
    unpack_state,
    update_bptt_critic,
)
from himloco_lab.rsl_rl.storage.amp_rollout_storage import AMPRolloutStorage
from himloco_lab.rsl_rl.storage.replay_buffer import ReplayBuffer
from himloco_lab.rsl_rl.storage.wm_replay_buffer import WMReplayBuffer


class AMPPPO:
    """AMP-PPO & SuperMimic BPTT hybrid algorithm."""

    actor_critic: AMPActorCritic

    def __init__(
        self,
        actor_critic: AMPActorCritic,
        discriminator: nn.Module,
        amp_data,
        amp_normalizer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 2.0e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        discriminator_lr: float = 1.0e-5,
        amp_replay_buffer_size: int = 1000000,
        min_std: torch.Tensor | None = None,
        **kwargs,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std
        self.extra_cfg = kwargs

        # AMP components
        self.discriminator = discriminator.to(self.device)
        self.amp_transition = AMPRolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # Actor-Critic components
        self.actor_critic = actor_critic.to(self.device)
        self.storage: AMPRolloutStorage | None = None

        # Optimizer for PPO + AMP Discriminator
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.disc_optimizer = optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-1},
            ],
            lr=discriminator_lr,
        )

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # SuperMimic BPTT components
        self.world_model: WorldModel | None = None
        self.bptt_critic: BPTTValueModel | None = None
        self.wm_replay: WMReplayBuffer | None = None
        self.wm_optimizer: optim.Optimizer | None = None
        self.bptt_actor_optimizer: optim.Optimizer | None = None
        self.bptt_critic_optimizer: optim.Optimizer | None = None
        self.bptt_horizon: int = int(kwargs.get("bptt_horizon", 32))
        self.num_one_step_obs: int = 45

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        critic_obs_shape: list[int],
        action_shape: list[int],
        amp_obs_shape: list[int] | None = None,
    ):
        del amp_obs_shape
        self.storage = AMPRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, amp_obs: torch.Tensor):
        self.amp_transition.observations = obs
        self.amp_transition.critic_observations = critic_obs
        self.amp_transition.amp_observations = amp_obs
        self.amp_transition.actions = self.actor_critic.act(obs).detach()
        self.amp_transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.amp_transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.amp_transition.actions
        ).detach()
        self.amp_transition.action_mean = self.actor_critic.action_mean.detach()
        self.amp_transition.action_sigma = self.actor_critic.action_std.detach()
        return self.amp_transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_amp_obs: torch.Tensor,
        next_critic_obs: torch.Tensor | None = None,
    ):
        self.amp_transition.rewards = rewards.clone()
        self.amp_transition.dones = dones
        self.amp_transition.next_amp_observations = next_amp_obs.clone()
        if next_critic_obs is not None:
            self.amp_transition.next_critic_observations = next_critic_obs.clone()
        elif hasattr(self.amp_transition, "critic_observations"):
            self.amp_transition.next_critic_observations = self.amp_transition.critic_observations.clone()

        if "time_outs" in infos:
            self.amp_transition.rewards += self.gamma * torch.squeeze(
                self.amp_transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.amp_storage.insert(self.amp_transition.amp_observations, self.amp_transition.next_amp_observations)
        self.storage.add_transitions(self.amp_transition)
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                next_critic_obs_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                *_,
            ) = sample

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if hasattr(self.actor_critic, "estimator") and next_critic_obs_batch is not None:
                estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch)
            else:
                estimation_loss, swap_loss = 0.0, 0.0

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.min_std is not None and hasattr(self.actor_critic, "std"):
                self.actor_critic.std.data = torch.max(self.actor_critic.std.data, self.min_std)

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += float(estimation_loss)
            mean_swap_loss += float(swap_loss)
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimation_loss,
            mean_swap_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
        )

    # ==========================================================================
    # SuperMimic BPTT 核心方法
    # ==========================================================================
    def init_supermimic_bptt(
        self,
        num_envs: int,
        num_joints: int = 12,
        num_one_step_obs: int = 45,
        default_joint_pos: torch.Tensor | None = None,
        joint_limits: torch.Tensor | None = None,
        wm_width: int = 2048,
        wm_lr: float = 2.0e-4,
        actor_lr: float = 2.0e-4,
        critic_lr: float = 2.0e-4,
        horizon: int = 32,
        wm_buffer_steps: int = 256,
        nominal_height: float = 0.72,
        min_height: float = 0.32,
        max_tilt_deg: float = 65.0,
        dt: float = 0.02,
    ):
        from ..modules.world_model import WorldModel, BPTTValueModel
        from ..storage.wm_replay_buffer import WMReplayBuffer

        self.bptt_horizon = int(horizon)
        self.num_one_step_obs = int(num_one_step_obs)

        self.world_model = WorldModel(
            joints=num_joints,
            frame_dim=num_one_step_obs,
            hidden=(wm_width, wm_width, wm_width),
            contact_dim=4,
            default_joint_pos=default_joint_pos,
            joint_limits=joint_limits,
            nominal_height=nominal_height,
            min_height=min_height,
            max_tilt_deg=max_tilt_deg,
            dt=dt,
        ).to(self.device)

        self.bptt_critic = BPTTValueModel(wm_inp_dim=self.world_model.inp_dim, command_dim=3).to(self.device)
        self.wm_replay = WMReplayBuffer(capacity=wm_buffer_steps, num_envs=num_envs, device=self.device)

        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=wm_lr)
        self.bptt_actor_optimizer = torch.optim.Adam(
            self.actor_critic.actor.parameters(), lr=actor_lr, betas=(0.9, 0.95)
        )
        self.bptt_critic_optimizer = torch.optim.Adam(self.bptt_critic.parameters(), lr=critic_lr)

    def update_world_model(
        self, wm_batches: int = 10, wm_batch_size: int = 2048, window_len: int = 8
    ) -> dict[str, float]:
        from ..modules.world_model import compute_wm_rollout_loss

        self.world_model.train()
        tot_loss, phys_loss, cont_loss, aux_loss = 0.0, 0.0, 0.0, 0.0
        actual_horizon = min(window_len, self.wm_replay.size)
        for _ in range(wm_batches):
            window = self.wm_replay.sample_prefix(batch_size=wm_batch_size, horizon=actual_horizon)
            loss, info = compute_wm_rollout_loss(self.world_model, window)
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.wm_optimizer.step()

            tot_loss += float(info["wm_loss"])
            phys_loss += float(info["wm_physics_loss"])
            cont_loss += float(info["wm_contact_loss"])
            aux_loss += float(info["wm_aux_loss"])

        n = max(wm_batches, 1)
        return {
            "wm_loss": tot_loss / n,
            "wm_physics_loss": phys_loss / n,
            "wm_contact_loss": cont_loss / n,
            "wm_aux_loss": aux_loss / n,
        }

    def update_bptt_step(
        self,
        batch_size: int = 1024,
        wm_batches: int = 10,
        wm_batch_size: int = 2048,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
    ) -> dict[str, float]:
        from ..modules.world_model import (
            freeze_parameters,
            critic_input,
            update_bptt_critic,
            unpack_state,
            quat_to_mat,
            mv,
        )

        # 1. 更新 8 步自回归世界模型
        wm_metrics = self.update_world_model(wm_batches=wm_batches, wm_batch_size=wm_batch_size)

        # 2. 从 WMReplayBuffer 采样真实转移数据更新 HIMEstimator
        starts = self.wm_replay.sample_starts(batch_size=batch_size)
        est_loss_val, swap_loss_val = 0.0, 0.0
        if "next_obs_history" in starts and "next_state" in starts:
            _, next_quat, next_lin_vel_w, _, _, _ = unpack_state(starts["next_state"], self.world_model.joints)
            next_Rt = quat_to_mat(next_quat).transpose(-1, -2)
            next_lin_vel_b = mv(next_Rt, next_lin_vel_w)
            if hasattr(self.actor_critic.estimator, "update_from_wm_replay"):
                est_loss_val, swap_loss_val = self.actor_critic.estimator.update_from_wm_replay(
                    starts["obs_history"], starts["next_obs_history"], next_lin_vel_b
                )

        # 3. 冻结 WorldModel、Critic、Discriminator 与 Estimator 权重，执行 H=32 步可微想象 BPTT 更新 Actor
        state = starts["state"]
        obs_history = starts["obs_history"]
        prev_action = starts["prev_action"]
        pre_pre_action = starts["pre_pre_action"]
        command = starts["command"][:, :3]

        critic_inputs_list = []
        rewards_list = []
        continuations_list = []

        self.actor_critic.actor.train()
        with freeze_parameters(
            self.world_model,
            self.bptt_critic,
            self.discriminator,
            self.actor_critic.estimator,
        ):
            self.bptt_actor_optimizer.zero_grad(set_to_none=True)
            for _ in range(self.bptt_horizon):
                critic_inputs_list.append(critic_input(self.world_model, state, prev_action, command))

                # 确定性可微策略前向
                action = self.actor_critic.act_bptt(obs_history)

                # 世界模型一步可微预测
                next_state, _, pred_rew, death_logit = self.world_model.predict(state, action)

                # 解析可微任务奖励与软存活率
                task_rew, cont = self.world_model.compute_differentiable_reward_and_continuation(
                    state=state,
                    next_state=next_state,
                    action=action,
                    prev_action=prev_action,
                    pre_pre_action=pre_pre_action,
                    command=command,
                    pred_reward=pred_rew,
                    death_logit=death_logit,
                )

                # 可微 42 维 AMP 观测与风格奖励融合
                if self.discriminator is not None:
                    amp_s = self.world_model.build_amp_obs(state)
                    amp_ns = self.world_model.build_amp_obs(next_state)
                    if hasattr(self.discriminator, "compute_differentiable_amp_reward"):
                        step_rew, _ = self.discriminator.compute_differentiable_amp_reward(
                            amp_s, amp_ns, task_rew, normalizer=self.amp_normalizer
                        )
                    else:
                        step_rew = task_rew
                else:
                    step_rew = task_rew

                # 构建下一时刻 45 维单步观测，并按 [最新帧在前, 历史帧在后] 推入 obs_history
                next_frame = self.world_model.build_actor_frame(next_state, action, command)
                if obs_history.shape[-1] > self.num_one_step_obs:
                    obs_history = torch.cat((next_frame, obs_history[:, :-self.num_one_step_obs]), dim=-1)
                else:
                    obs_history = next_frame

                rewards_list.append(step_rew)
                continuations_list.append(cont)

                pre_pre_action = prev_action
                prev_action = action
                state = next_state

            # 末端第 H+1 步的 Critic 输入与 Bootstrap 价值
            critic_inputs_list.append(critic_input(self.world_model, state, prev_action, command))
            critic_inputs = torch.stack(critic_inputs_list, dim=0)  # [H+1, B, D]
            rewards = torch.stack(rewards_list, dim=0)              # [H, B]
            continuations = torch.stack(continuations_list, dim=0)  # [H, B]

            # 对齐 SuperMimic 的累积存活率加权折扣回报 + H 步 Bootstrap
            alive = torch.cumprod(continuations, dim=0)
            weights = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), dim=0)
            discounts = (self.gamma ** torch.arange(self.bptt_horizon, device=self.device, dtype=rewards.dtype))[:, None]
            imagined_return = (discounts * weights * rewards).sum(dim=0)
            bootstrap_value = (self.gamma ** self.bptt_horizon) * alive[-1] * self.bptt_critic(critic_inputs[-1])
            objective = (imagined_return + bootstrap_value).mean()

            actor_loss = -objective
            actor_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_value_(self.actor_critic.actor.parameters(), 0.1)
            self.bptt_actor_optimizer.step()

        # 4. 使用脱离计算图的想象轨迹与 GAE 目标更新 BPTT Critic
        rollout_data = {
            "critic_inputs": critic_inputs.detach(),
            "rewards": rewards.detach(),
            "continuations": continuations.detach(),
        }
        critic_loss = update_bptt_critic(
            critic=self.bptt_critic,
            optimizer=self.bptt_critic_optimizer,
            rollout=rollout_data,
            epochs=critic_epochs,
            minibatches=critic_minibatches,
            gamma=self.gamma,
            lam=self.lam,
        )

        return {
            **wm_metrics,
            "bptt_critic_loss": float(critic_loss.detach()),
            "estimator_loss": float(est_loss_val),
            "swap_loss": float(swap_loss_val),
            "bptt_imagined_return": float(imagined_return.detach().mean()),
            "bptt_survival_rate": float(alive[-1].detach().mean()),
            "bptt_actor_grad_norm": float(grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        }

import torch
import torch.nn as nn
import torch.optim as optim

from ..modules.amp_actor_critic import AMPActorCritic
from ..storage.amp_rollout_storage import AMPRolloutStorage
from ..storage.replay_buffer import ReplayBuffer

class AMPPPO:
    actor_critic: AMPActorCritic
    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 min_std=None,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = AMPRolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        # Optimizer for policy and discriminator.
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {'params': self.discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 'name': 'amp_head'}]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = AMPRolloutStorage.Transition()

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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = AMPRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs,amp_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs,amp_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.amp_storage.insert(
            self.amp_transition.observations, amp_obs)
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
        
                obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch = sample
                
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Discriminator loss.
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
                expert_loss = torch.nn.MSELoss()(
                    expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(
                    policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    expert_state, expert_next_state, lambda_=10)
                
                # Compute total loss.
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    amp_loss + grad_pen_loss)
                # loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                    self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
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

        return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss,mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred
    
    def init_supermimic_world_model(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        wm_state_to_amp_obs_fn,
        cmd_dim: int = 3,
        wm_lr: float = 2e-4,
        bptt_horizon: int = 32,
        wm_window_len: int = 8,
    ):
        from himloco_lab.rsl_rl.modules.world_model import DifferentiableWorldModel

        self.bptt_horizon = bptt_horizon
        self.wm_window_len = wm_window_len
        self.wm_state_to_amp_obs_fn = wm_state_to_amp_obs_fn

        self.world_model = DifferentiableWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            context_dim=context_dim,
            cmd_dim=cmd_dim,
            device=self.device,
        ).to(self.device)

        wm_params = (
            list(self.world_model.dynamics_net.parameters())
            + list(self.world_model.reward_head.parameters())
            + list(self.world_model.term_head.parameters())
        )
        self.wm_optimizer = torch.optim.Adam(wm_params, lr=wm_lr, betas=(0.9, 0.95))
        self.bptt_actor_optimizer = torch.optim.Adam(
            self.actor_critic.actor.parameters(), lr=wm_lr, betas=(0.9, 0.95)
        )
        self.bptt_critic_optimizer = torch.optim.Adam(
            self.world_model.wm_critic.parameters(), lr=wm_lr, betas=(0.9, 0.95)
        )
        # 独立的判别器优化器，避免触碰 self.optimizer 中的 actor_critic 动量
        self.bptt_disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=wm_lr, betas=(0.9, 0.95)
        )

    def update_world_model(self, wm_buffer, num_batches: int = 10, batch_size: int = 2048) -> dict[str, float]:
        self.world_model.train()
        agg_metrics = {
            "wm_loss_total": 0.0,
            "wm_loss_state": 0.0,
            "wm_loss_reward": 0.0,
            "wm_loss_done": 0.0,
            "amp_disc_loss": 0.0,
            "amp_grad_pen": 0.0,
        }

        expert_gen = self.amp_data.feed_forward_generator(num_batches, batch_size)

        for _ in range(num_batches):
            (
                s_seq, a_seq, o_seq, r_seq, falls_seq, resets_seq, cmd_seq, xi_seq
            ) = wm_buffer.sample_k_step_windows(batch_size=batch_size, k=self.wm_window_len)

            self.world_model.state_norm.update(s_seq.reshape(-1, s_seq.shape[-1]))
            valid_deltas = (s_seq[:, 1:, :] - s_seq[:, :-1, :])[~resets_seq]
            if valid_deltas.numel() > 0:
                self.world_model.delta_norm.update(valid_deltas.reshape(-1, s_seq.shape[-1]))
            self.world_model.context_norm.update(xi_seq.reshape(-1, xi_seq.shape[-1]))

            loss, metrics = self.world_model.compute_k_step_loss(
                s_seq, a_seq, o_seq, r_seq, falls_seq, resets_seq, cmd_seq, xi_seq
            )
            self.wm_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.wm_optimizer.step()

            for k_name, v in metrics.items():
                agg_metrics[k_name] += v / num_batches

            valid_trans = ~resets_seq[:, 0]
            if valid_trans.any() and self.amp_data is not None:
                policy_s = self.wm_state_to_amp_obs_fn(s_seq[valid_trans, 0, :]).detach()
                policy_next_s = self.wm_state_to_amp_obs_fn(s_seq[valid_trans, 1, :]).detach()
                expert_s, expert_next_s = next(expert_gen)
                n_min = min(policy_s.shape[0], expert_s.shape[0])
                policy_s, policy_next_s = policy_s[:n_min], policy_next_s[:n_min]
                expert_s, expert_next_s = expert_s[:n_min], expert_next_s[:n_min]

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_s.cpu().numpy())
                    self.amp_normalizer.update(expert_s.cpu().numpy())
                    policy_s = self.amp_normalizer.normalize_torch(policy_s, self.device)
                    policy_next_s = self.amp_normalizer.normalize_torch(policy_next_s, self.device)
                    expert_s = self.amp_normalizer.normalize_torch(expert_s, self.device)
                    expert_next_s = self.amp_normalizer.normalize_torch(expert_next_s, self.device)

                expert_d = self.discriminator(torch.cat([expert_s, expert_next_s], dim=-1))
                policy_d = self.discriminator(torch.cat([policy_s, policy_next_s], dim=-1))

                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
                policy_loss = torch.nn.MSELoss()(policy_d, -1.0 * torch.ones_like(policy_d))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen = self.discriminator.compute_grad_pen(expert_s, expert_next_s, lambda_=10.0)

                self.bptt_disc_optimizer.zero_grad()
                (amp_loss + grad_pen).backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.bptt_disc_optimizer.step()

                agg_metrics["amp_disc_loss"] += amp_loss.item() / num_batches
                agg_metrics["amp_grad_pen"] += grad_pen.item() / num_batches

        return agg_metrics

    def sample_sensor_noise(self, clean_obs: torch.Tensor) -> torch.Tensor:
        """严格对齐 ecow PolicyCfg 的噪声与缩放系数 (ang_vel scale=0.25, joint_vel scale=0.05)"""
        noise = torch.zeros_like(clean_obs)
        noise[..., 3:6].uniform_(-0.2 * 0.25, 0.2 * 0.25)
        noise[..., 6:9].uniform_(-0.05, 0.05)
        noise[..., 9:21].uniform_(-0.01, 0.01)
        noise[..., 21:33].uniform_(-1.5 * 0.05, 1.5 * 0.05)
        return clean_obs + noise

    def update_bptt(self, wm_buffer, batch_size: int = 4096) -> dict[str, float]:
        H = self.bptt_horizon
        for p in self.world_model.parameters():
            p.requires_grad_(False)
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        if hasattr(self.actor_critic, "estimator"):
            for p in self.actor_critic.estimator.parameters():
                p.requires_grad_(False)

        s_t, clean_obs_hist_t, cmd, xi = wm_buffer.sample_initial_states(batch_size=batch_size)
        noisy_obs_hist_t = self.sample_sensor_noise(clean_obs_hist_t)

        survival_c = torch.ones(batch_size, device=self.device)
        discount = 1.0

        imagined_states, imagined_commands, imagined_contexts = [], [], []
        imagined_rewards, imagined_dones = [], []
        policy_return = torch.zeros(batch_size, device=self.device)

        lerp = getattr(self.discriminator, "task_reward_lerp", 0.7)
        reward_coef = getattr(self.discriminator, "reward_coef", 0.05)

        for t in range(H):
            imagined_states.append(s_t)
            imagined_commands.append(cmd)
            imagined_contexts.append(xi)

            curr_obs_hist_flat = noisy_obs_hist_t.reshape(batch_size, -1)
            prev_action = noisy_obs_hist_t[:, 0, 33:45]

            if hasattr(self.actor_critic, "estimator") and hasattr(
                self.actor_critic.estimator, "forward_with_grad"
            ):
                vel, latent = self.actor_critic.estimator.forward_with_grad(curr_obs_hist_flat)
                raw_actions = self.actor_critic.actor(
                    torch.cat([curr_obs_hist_flat[:, :45], vel, latent], dim=-1)
                )
            else:
                raw_actions = self.actor_critic.act_inference(curr_obs_hist_flat)

            actions = 10.0 * torch.tanh(raw_actions / 10.0)

            next_s_t, next_clean_obs, pred_task_rew, _, pred_done_prob = self.world_model.forward_step(
                s_t, actions, prev_action, xi, cmd
            )

            # 从 46 维统一状态可微重构 42 维 AMP 状态
            amp_s = self.wm_state_to_amp_obs_fn(s_t)
            amp_next_s = self.wm_state_to_amp_obs_fn(next_s_t)
            if self.amp_normalizer is not None:
                amp_s_norm = self.amp_normalizer.normalize_torch(amp_s, self.device)
                amp_next_s_norm = self.amp_normalizer.normalize_torch(amp_next_s, self.device)
            else:
                amp_s_norm, amp_next_s_norm = amp_s, amp_next_s

            disc_logits = self.discriminator(torch.cat([amp_s_norm, amp_next_s_norm], dim=-1))
            # Leaky-LSGAN：消除 disc_logits <= -1.0 时的 clamp(min=0) 梯度死区，持续提供迈步风格梯度！
            raw_amp_rew = 1.0 - 0.25 * torch.square(disc_logits - 1.0)
            amp_rew = torch.where(raw_amp_rew >= 0.0, raw_amp_rew, 0.25 * raw_amp_rew).squeeze(-1)
            is_moving = (torch.norm(cmd[:, :3], dim=-1) >= 0.1).float()
            amp_rew = amp_rew * is_moving * reward_coef

            step_reward = (1.0 - lerp) * amp_rew + lerp * pred_task_rew
            policy_return = policy_return + discount * survival_c * step_reward

            imagined_rewards.append(step_reward)
            imagined_dones.append(pred_done_prob)

            survival_c = survival_c * (1.0 - pred_done_prob)
            discount *= self.gamma

            noisy_next_obs = self.sample_sensor_noise(next_clean_obs)
            noisy_obs_hist_t = torch.cat(
                [noisy_next_obs.unsqueeze(1), noisy_obs_hist_t[:, :-1, :]], dim=1
            )
            s_t = next_s_t

        terminal_value = self.world_model.evaluate_critic(s_t, cmd, xi).squeeze(-1)
        policy_return = policy_return + discount * survival_c * terminal_value

        actor_loss = -policy_return.mean()
        self.bptt_actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_norm=1.0)
        for p in self.actor_critic.actor.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-0.1, 0.1)
        self.bptt_actor_optimizer.step()

        for p in self.world_model.parameters():
            p.requires_grad_(True)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        if hasattr(self.actor_critic, "estimator"):
            for p in self.actor_critic.estimator.parameters():
                p.requires_grad_(True)

        critic_loss_val = self._update_critic_on_imagined(
            imagined_states, imagined_commands, imagined_contexts,
            imagined_rewards, imagined_dones, terminal_value.detach()
        )

        return {
            "bptt_actor_return": (-actor_loss).item(),
            "bptt_critic_loss": critic_loss_val,
            "bptt_mean_survival_H": survival_c.mean().item(),
        }

    def _update_critic_on_imagined(
        self, states_list, cmds_list, xis_list, rewards_list, dones_list, v_H: torch.Tensor
    ) -> float:
        import torch.nn.functional as F

        H = len(rewards_list)
        B = v_H.shape[0]

        with torch.no_grad():
            states_stack = torch.stack([s.detach() for s in states_list], dim=0)
            cmds_stack = torch.stack([c.detach() for c in cmds_list], dim=0)
            xis_stack = torch.stack([x.detach() for x in xis_list], dim=0)
            rewards_stack = torch.stack([r.detach() for r in rewards_list], dim=0)
            dones_stack = torch.stack([d.detach() for d in dones_list], dim=0)

            flat_s = states_stack.reshape(H * B, -1)
            flat_c = cmds_stack.reshape(H * B, -1)
            flat_x = xis_stack.reshape(H * B, -1)
            values = self.world_model.evaluate_critic(flat_s, flat_c, flat_x).reshape(H, B)

            returns = torch.zeros_like(values)
            R_next = v_H
            v_next = v_H
            for t in reversed(range(H)):
                R_curr = rewards_stack[t] + self.gamma * (1.0 - dones_stack[t]) * (
                    (1.0 - self.lam) * v_next + self.lam * R_next
                )
                returns[t] = R_curr
                R_next = R_curr
                v_next = values[t]

            flat_targets = returns.reshape(H * B, 1).detach()

        total_samples = H * B
        minibatch_size = total_samples // 4
        total_v_loss = 0.0
        for _ in range(5):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                v_pred = self.world_model.evaluate_critic(flat_s[mb_idx], flat_c[mb_idx], flat_x[mb_idx])
                v_loss = F.mse_loss(v_pred, flat_targets[mb_idx])

                self.bptt_critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.world_model.wm_critic.parameters(), max_norm=1.0)
                self.bptt_critic_optimizer.step()
                total_v_loss += v_loss.item()

        return total_v_loss / (5 * 4)


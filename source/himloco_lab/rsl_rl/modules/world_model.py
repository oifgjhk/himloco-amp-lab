# source/himloco_lab/rsl_rl/modules/world_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd(nn.Module):
    """纯 PyTorch GPU 在线均值方差归一化器"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=1e-4))
        self.count.copy_(tot_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class DifferentiableWorldModel(nn.Module):
    """SuperMimic 可微世界模型 (Section 3.2 & Table A.4 / Table A.6)"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        cmd_dim: int = 3,
        hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        dt: float = 0.02,
        device: str = "cuda",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.cmd_dim = cmd_dim
        self.dt = dt
        self.device = device

        self.state_norm = RunningMeanStd(state_dim).to(device)
        self.context_norm = RunningMeanStd(context_dim).to(device)

        # 1. 核心动力学网络 f_omega(s_t, a_t, xi) -> delta_s (MLP: 2048 x 3)
        dyn_in_dim = state_dim + action_dim + context_dim
        layers = []
        in_d = dyn_in_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.SiLU()])
            in_d = h_d
        final_dyn = nn.Linear(in_d, state_dim)
        nn.init.uniform_(final_dyn.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_dyn.bias)
        layers.append(final_dyn)
        self.dynamics_net = nn.Sequential(*layers)

        # 2. 残差任务奖励头 r_omega_res(s_{t+1}, a_t, a_{t-1}, g_t)
        rew_in_dim = state_dim + action_dim * 2 + cmd_dim
        final_rew = nn.Linear(256, 1)
        nn.init.uniform_(final_rew.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_rew.bias)
        self.reward_head = nn.Sequential(
            nn.Linear(rew_in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            final_rew,
        )

        # 3. 软终止头 d_omega(s_{t+1}, g_t) (偏置初始化为 -4.0，使初始死亡率约为 0.018 而非 0.5)
        final_term = nn.Linear(128, 1)
        nn.init.uniform_(final_term.weight, -1e-2, 1e-2)
        nn.init.constant_(final_term.bias, -4.0)
        self.term_head = nn.Sequential(
            nn.Linear(state_dim + cmd_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            final_term,
        )

        # 4. 想象滚动专用 Critic V_psi(s_t, g_t, xi) (Table A.4: MLP 256, 128)
        self.wm_critic = nn.Sequential(
            nn.Linear(state_dim + cmd_dim + context_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def evaluate_critic(self, state: torch.Tensor, cmd: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        critic_in = torch.cat([self.state_norm(state), cmd, self.context_norm(context)], dim=-1)
        return self.wm_critic(critic_in)

    def compute_analytic_task_reward(
        self,
        next_state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """解析主任务奖励（提供精确的速度跟踪、水平姿态与动作平滑一阶梯度）"""
        lin_vel_b = next_state[:, 1:4]
        ang_vel_b = next_state[:, 4:7]
        proj_grav = next_state[:, 7:10]

        lin_vel_err = torch.sum(torch.square(command[:, :2] - lin_vel_b[:, :2]), dim=-1)
        rew_lin_vel = 2.5 * torch.exp(-lin_vel_err / 0.1)

        ang_vel_err = torch.square(command[:, 2] - ang_vel_b[:, 2])
        rew_ang_vel = 1.5 * torch.exp(-ang_vel_err / 0.1)

        pen_lin_z = -2.0 * torch.square(lin_vel_b[:, 2])
        pen_ang_xy = -0.25 * torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1)
        pen_flat = -1.5 * torch.sum(torch.square(proj_grav[:, :2]), dim=-1)
        pen_action_rate = -0.03 * torch.sum(torch.square(action - prev_action), dim=-1)
        pen_action_l2 = -0.002 * torch.sum(torch.square(action), dim=-1)

        analytic_rew = (
            rew_lin_vel + rew_ang_vel + pen_lin_z + pen_ang_xy + pen_flat + pen_action_rate + pen_action_l2
        ) * self.dt
        return analytic_rew

    def forward_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        context: torch.Tensor,
        command: torch.Tensor,
    ):
        norm_s = self.state_norm(state)
        norm_xi = self.context_norm(context)

        dyn_in = torch.cat([norm_s, action, norm_xi], dim=-1)
        delta_s = self.dynamics_net(dyn_in)
        next_state = state + delta_s

        # 单位化 projected_gravity (索引 7:10)
        grav = next_state[:, 7:10]
        next_state = torch.cat(
            [
                next_state[:, :7],
                grav / (torch.norm(grav, dim=-1, keepdim=True) + 1e-6),
                next_state[:, 10:],
            ],
            dim=-1,
        )

        # 解析观测头 g_omega(\hat{s}_{t+1}): 重构 45 维无噪观测 \hat{\bar{o}}_{t+1}
        base_ang_vel = next_state[:, 4:7] * 0.25
        proj_gravity = next_state[:, 7:10]
        dof_pos_rel = next_state[:, 10:22]
        dof_vel_rel = next_state[:, 22:34] * 0.05
        clean_next_obs = torch.cat(
            [command, base_ang_vel, proj_gravity, dof_pos_rel, dof_vel_rel, action], dim=-1
        )

        analytic_rew = self.compute_analytic_task_reward(next_state, action, prev_action, command)
        rew_in = torch.cat([self.state_norm(next_state), action, prev_action, command], dim=-1)
        res_rew = self.reward_head(rew_in).squeeze(-1)
        pred_reward = analytic_rew + res_rew

        term_in = torch.cat([self.state_norm(next_state), command], dim=-1)
        pred_done_logits = self.term_head(term_in).squeeze(-1)
        pred_done_prob = torch.sigmoid(pred_done_logits)

        return next_state, clean_next_obs, pred_reward, pred_done_logits, pred_done_prob

    def compute_k_step_loss(
        self,
        states_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        clean_obs_seq: torch.Tensor,
        rewards_seq: torch.Tensor,
        falls_seq: torch.Tensor,
        resets_seq: torch.Tensor,
        commands_seq: torch.Tensor,
        context_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        k = actions_seq.shape[1]
        pred_s = states_seq[:, 0, :]
        prev_act = clean_obs_seq[:, 0, 33:45]

        loss_s, loss_o, loss_r, loss_d = 0.0, 0.0, 0.0, 0.0
        valid_step_mask = torch.ones(states_seq.shape[0], device=self.device)

        for j in range(k):
            curr_act = actions_seq[:, j, :]
            pred_s, pred_o, pred_r, pred_d_logits, _ = self.forward_step(
                pred_s, curr_act, prev_act, context_seq[:, j, :], commands_seq[:, j, :]
            )
            prev_act = curr_act

            loss_d_step = F.binary_cross_entropy_with_logits(
                pred_d_logits, falls_seq[:, j].float(), reduction="none"
            )
            loss_d = loss_d + (loss_d_step * valid_step_mask).mean()

            not_reset_j = (1.0 - resets_seq[:, j].float()) * valid_step_mask
            valid_count = torch.clamp(not_reset_j.sum(), min=1.0)

            step_mse_s = torch.mean(torch.square(pred_s - states_seq[:, j + 1, :]), dim=-1)
            step_mse_o = torch.mean(torch.square(pred_o - clean_obs_seq[:, j + 1, :]), dim=-1)
            step_mse_r = torch.square(pred_r - rewards_seq[:, j])

            loss_s = loss_s + (step_mse_s * not_reset_j).sum() / valid_count
            loss_o = loss_o + (step_mse_o * not_reset_j).sum() / valid_count
            loss_r = loss_r + (step_mse_r * not_reset_j).sum() / valid_count

            valid_step_mask = not_reset_j

        total_loss = (loss_s + loss_o + loss_r + loss_d) / k
        return total_loss, {
            "wm_loss_total": total_loss.item(),
            "wm_loss_state": (loss_s / k).item(),
            "wm_loss_reward": (loss_r / k).item(),
            "wm_loss_done": (loss_d / k).item(),
        }
# source/himloco_lab/rsl_rl/modules/world_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd(nn.Module):
    """纯 PyTorch GPU 在线均值方差归一化器，支持可微前向传播"""

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
    """SuperMimic 可微世界模型 (Section 3.2 & Table A.4)"""

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

        # 1. 动力学网络 f_omega(s_t, a_t, xi) -> delta_s (MLP: 2048 x 3, Table A.4)
        dyn_in_dim = state_dim + action_dim + context_dim
        layers = []
        in_d = dyn_in_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.SiLU()])
            in_d = h_d
        layers.append(nn.Linear(in_d, state_dim))
        self.dynamics_net = nn.Sequential(*layers)

        # 2. 任务奖励头 r_omega(s_t, a_t, g_t)
        rew_in_dim = state_dim + action_dim + cmd_dim
        self.reward_head = nn.Sequential(
            nn.Linear(rew_in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

        # 3. 软终止头 d_omega(s_t, g_t)
        self.term_head = nn.Sequential(
            nn.Linear(state_dim + cmd_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # 4. 想象滚动专用 Critic V_psi(s_t, g_t, xi) (对照论文 Table A.4: MLP 256, 128)
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

    def forward_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
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

        # 解析观测头：从 next_state 前 34 维重构 45 维无噪观测 \hat{\bar{o}}_{t+1}
        base_ang_vel = next_state[:, 4:7] * 0.25
        proj_gravity = next_state[:, 7:10]
        dof_pos_rel = next_state[:, 10:22]
        dof_vel_rel = next_state[:, 22:34] * 0.05
        clean_next_obs = torch.cat(
            [command, base_ang_vel, proj_gravity, dof_pos_rel, dof_vel_rel, action], dim=-1
        )

        rew_in = torch.cat([self.state_norm(next_state), action, command], dim=-1)
        pred_reward = self.reward_head(rew_in).squeeze(-1)

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
        dones_seq: torch.Tensor,
        commands_seq: torch.Tensor,
        context_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        k = actions_seq.shape[1]
        pred_s = states_seq[:, 0, :]

        loss_s, loss_o, loss_r, loss_d = 0.0, 0.0, 0.0, 0.0
        for j in range(k):
            pred_s, pred_o, pred_r, pred_d_logits, _ = self.forward_step(
                pred_s, actions_seq[:, j, :], context_seq[:, j, :], commands_seq[:, j, :]
            )
            loss_s = loss_s + F.mse_loss(pred_s, states_seq[:, j + 1, :])
            loss_o = loss_o + F.mse_loss(pred_o, clean_obs_seq[:, j + 1, :])
            loss_r = loss_r + F.mse_loss(pred_r, rewards_seq[:, j])
            loss_d = loss_d + F.binary_cross_entropy_with_logits(pred_d_logits, dones_seq[:, j].float())

        total_loss = (loss_s + loss_o + loss_r + loss_d) / k
        return total_loss, {
            "wm_loss_total": total_loss.item(),
            "wm_loss_state": (loss_s / k).item(),
            "wm_loss_reward": (loss_r / k).item(),
            "wm_loss_done": (loss_d / k).item(),
        }
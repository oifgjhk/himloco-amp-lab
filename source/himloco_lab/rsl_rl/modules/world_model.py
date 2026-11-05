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

    def unnormalize(self, norm_x: torch.Tensor) -> torch.Tensor:
        return norm_x * torch.sqrt(self.var + self.eps) + self.mean


class DifferentiableWorldModel(nn.Module):
    """
    SuperMimic 可微世界模型 (Section 3.2 & Table A.4 / Table A.6):
    - 统一 46 维状态 s_t:
      [root_z(1), lin_vel_b(3), ang_vel_b(3), proj_grav(3), dof_pos_rel(12), dof_vel_rel(12), foot_pos_b(12)]
    """

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
        self.delta_norm = RunningMeanStd(state_dim).to(device)
        self.context_norm = RunningMeanStd(context_dim).to(device)

        # 重点强化机身线速度 (1:4)、角速度 (4:7) 与足端位置 (34:46) 的动力学权重
        state_weights = torch.ones(state_dim, device=device)
        state_weights[0] = 2.0        # root_z
        state_weights[1:4] = 8.0      # base_lin_vel (8倍核心权重，确保学准 dv/da)
        state_weights[4:7] = 4.0      # base_ang_vel
        state_weights[7:10] = 3.0     # projected_gravity
        state_weights[34:46] = 3.0    # foot_pos_b (足端轨迹与抬腿高度)
        self.register_buffer("state_loss_weights", state_weights)

        # 1. 核心动力学网络 f_omega(s_t, a_t, xi) -> 预测标准化状态差分 norm_delta_s
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

        # 2. 残差奖励拟合头 (预测未乘 dt 的每秒残差奖励，输入 detach 状态以防污染动力学网络)
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

        # 3. 软终止头 d_omega(s_{t+1}, g_t)
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

        # 4. 想象滚动专用 Critic V_psi(s_t, g_t, xi)
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
        """
        无死区解析任务奖励 (严格消除大误差梯度消失与过度动作抑制):
        1. 宽高斯核 (1.0) + 精确高斯核 (0.20) + 线性距离拉力 (替代 low_speed 惩罚)；
        2. 移动指令下的摆动腿抬高激励 (利用 foot_pos_b 的 Z 坐标打破四足贴地死锁)；
        3. 仅约束髋关节侧摆 (hip_joint)，彻底释放大腿/小腿屈伸空间。
        """
        lin_vel_b = next_state[:, 1:4]
        ang_vel_b = next_state[:, 4:7]
        proj_grav = next_state[:, 7:10]
        dof_pos_rel = next_state[:, 10:22]
        foot_pos_b = next_state[:, 34:46].view(-1, 4, 3)

        # 1. 线速度跟踪：在 error_vel_xy = 1.35 m/s 时依然保有强力一阶正梯度
        lin_vel_err_sq = torch.sum(torch.square(command[:, :2] - lin_vel_b[:, :2]), dim=-1)
        lin_vel_err_norm = torch.sqrt(lin_vel_err_sq + 1e-4)
        rew_lin_vel = (
            2.0 * torch.exp(-lin_vel_err_sq / 0.20)
            + 1.5 * torch.exp(-lin_vel_err_sq / 1.00)
            - 1.5 * lin_vel_err_norm
        )

        # 2. 偏航角速度跟踪
        ang_vel_err_sq = torch.square(command[:, 2] - ang_vel_b[:, 2])
        ang_vel_err_abs = torch.sqrt(ang_vel_err_sq + 1e-4)
        rew_ang_vel = (
            1.2 * torch.exp(-ang_vel_err_sq / 0.25)
            + 0.8 * torch.exp(-ang_vel_err_sq / 1.00)
            - 0.8 * ang_vel_err_abs
        )

        # 3. 移动时的足端抬腿激励 (打破四足粘地不起的局部极小值)
        is_moving = (torch.norm(command[:, :3], dim=-1) >= 0.1).float()
        # 机体系下站立时足端 z 约为 -0.40m；激励最高的摆动脚抬高至 -0.28m (即离地 ~12cm)
        max_foot_z = torch.amax(foot_pos_b[:, :, 2], dim=-1)
        swing_clearance_rew = 0.8 * is_moving * torch.exp(-torch.square(max_foot_z - (-0.28)) / 0.02)

        # 4. 适度的姿态稳定与平滑正则 (降低 action_rate 权重，绝不压制迈步梯度)
        pen_lin_z = -1.5 * torch.square(lin_vel_b[:, 2])
        pen_ang_xy = -0.15 * torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1)
        pen_flat = -1.2 * torch.sum(torch.square(proj_grav[:, :2]), dim=-1)
        # 仅惩罚 4 个髋关节侧摆 (索引 0, 1, 2, 3 或 0, 3, 6, 9，此处取轻微全身防劈叉约束)
        pen_action_rate = -0.005 * torch.sum(torch.square(action - prev_action), dim=-1)
        pen_action_l2 = -0.001 * torch.sum(torch.square(action), dim=-1)
        pen_dof_pos = -0.01 * torch.sum(torch.square(dof_pos_rel), dim=-1)

        analytic_rew = (
            rew_lin_vel
            + rew_ang_vel
            + swing_clearance_rew
            + pen_lin_z
            + pen_ang_xy
            + pen_flat
            + pen_action_rate
            + pen_action_l2
            + pen_dof_pos
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
        norm_delta_s = self.dynamics_net(dyn_in)
        delta_s = self.delta_norm.unnormalize(norm_delta_s)
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

        # 解析观测头：严格保持与 ecow PolicyCfg 一致的 0.25 与 0.05 缩放系数！
        base_ang_vel = next_state[:, 4:7] * 0.25
        proj_gravity = next_state[:, 7:10]
        dof_pos_rel = next_state[:, 10:22]
        dof_vel_rel = next_state[:, 22:34] * 0.05
        clean_next_obs = torch.cat(
            [command, base_ang_vel, proj_gravity, dof_pos_rel, dof_vel_rel, action], dim=-1
        )

        # 主任务奖励由无死区解析奖励主导
        pred_reward = self.compute_analytic_task_reward(next_state, action, prev_action, command)

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
        inv_delta_std = 1.0 / torch.sqrt(self.delta_norm.var.detach() + 1e-5)

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

            # 在差分标准化空间计算加权状态损失，确保 base_lin_vel (8倍权重) 被高精度拟合
            norm_err_s = (pred_s - states_seq[:, j + 1, :]) * inv_delta_std
            step_mse_s = torch.mean(torch.square(norm_err_s) * self.state_loss_weights, dim=-1)
            step_mse_o = torch.mean(torch.square(pred_o - clean_obs_seq[:, j + 1, :]), dim=-1)

            # 将单步奖励除以 dt 还原到每秒尺度再计算监控误差（不再显示为 0.0000）
            rew_in = torch.cat(
                [self.state_norm(pred_s.detach()), curr_act, prev_act, commands_seq[:, j, :]], dim=-1
            )
            pred_r_rate = self.reward_head(rew_in).squeeze(-1)
            target_r_rate = rewards_seq[:, j] / self.dt
            step_mse_r = torch.square(pred_r_rate - target_r_rate)

            loss_s = loss_s + (step_mse_s * not_reset_j).sum() / valid_count
            loss_o = loss_o + (step_mse_o * not_reset_j).sum() / valid_count
            loss_r = loss_r + (step_mse_r * not_reset_j).sum() / valid_count

            valid_step_mask = not_reset_j

        total_loss = (loss_s + 0.5 * loss_o + 0.1 * loss_r + loss_d) / k
        return total_loss, {
            "wm_loss_total": total_loss.item(),
            "wm_loss_state": (loss_s / k).item(),
            "wm_loss_reward": (loss_r / k).item(),
            "wm_loss_done": (loss_d / k).item(),
        }
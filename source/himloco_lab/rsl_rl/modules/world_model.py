# Copyright (c) 2025, HimLoco Lab Project Developers.
# Reference: SuperMimic (supermimic/core/world_model.py, state.py, rotations.py, imagination.py, velocity.py)

from __future__ import annotations

from contextlib import contextmanager
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def freeze_parameters(*modules: nn.Module):
    """在 BPTT 期间冻结世界模型、Critic、Estimator 和 Discriminator 的权重更新，但保留对输入的梯度计算"""
    original = [(p, p.requires_grad) for m in modules if m is not None for p in m.parameters()]
    try:
        for p, _ in original:
            p.requires_grad_(False)
        yield
    finally:
        for p, val in original:
            p.requires_grad_(val)


# ==============================================================================
# 1. 旋转与流形工具函数 (wxyz 四元数约定，严格对齐 supermimic/core/rotations.py)
# ==============================================================================
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    cands = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half < 1e-6, 2.0 / torch.clamp(q[..., :1], min=1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_log(R: torch.Tensor) -> torch.Tensor:
    return quat_to_rotvec(mat_to_quat(R))


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (matrix * vector.unsqueeze(-2)).sum(-1)


# ==============================================================================
# 2. 物理状态打包/解包与局部增量操作 (严格对齐 supermimic/core/state.py)
# ==============================================================================
def unpack_state(x: torch.Tensor, joints: int = 12):
    return (
        x[..., :3],               # root_pos (3)
        x[..., 3:7],              # root_quat_wxyz (4)
        x[..., 7:10],             # root_lin_vel_w (3)
        x[..., 10:13],            # root_ang_vel_w (3)
        x[..., 13:13 + joints],   # joint_pos (J)
        x[..., 13 + joints:],     # joint_vel (J)
    )


def state_features(x: torch.Tensor, joints: int = 12) -> torch.Tensor:
    pos, quat, lin_vel_w, ang_vel_w, q, dq = unpack_state(x, joints)
    R = quat_to_mat(quat)
    Rt = R.transpose(-1, -2)
    heading = yaw_mat(R)
    return torch.cat(
        (
            pos[..., 2:3],
            mat_to_6d(heading.transpose(-1, -2) @ R),
            mv(Rt, lin_vel_w),
            mv(Rt, ang_vel_w),
            q,
            dq,
        ),
        dim=-1,
    )


def state_delta(source: torch.Tensor, target: torch.Tensor, joints: int = 12) -> torch.Tensor:
    sp, sq, sv, sw, sj, sdj = unpack_state(source, joints)
    tp, tq, tv, tw, tj, tdj = unpack_state(target, joints)
    R = quat_to_mat(sq)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            mv(yaw_mat(R).transpose(-1, -2), tp - sp),
            so3_log(Rt @ quat_to_mat(tq)),
            mv(Rt, tv - sv),
            mv(Rt, tw - sw),
            tj - sj,
            tdj - sdj,
        ),
        dim=-1,
    )


def state_integrate(x: torch.Tensor, increments: torch.Tensor, joints: int = 12) -> torch.Tensor:
    pos, quat, lin_vel_w, ang_vel_w, q, dq = unpack_state(x, joints)
    dp, dr, dv, dw, dq_inc, ddq_inc = increments.split((3, 3, 3, 3, joints, joints), dim=-1)
    R = quat_to_mat(quat)
    return torch.cat(
        (
            pos + mv(yaw_mat(R), dp),
            mat_to_quat(R @ so3_exp(dr)),
            lin_vel_w + mv(R, dv),
            ang_vel_w + mv(R, dw),
            q + dq_inc,
            dq + ddq_inc,
        ),
        dim=-1,
    )


# ==============================================================================
# 3. RunningNorm 在线归一化模块 (对齐 supermimic/core/normalization.py)
# ==============================================================================
class RunningNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2, max_count: float = 2e8):
        super().__init__()
        self.eps = eps
        self.max_count = max_count
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if float(self.count) >= self.max_count or not x.numel():
            return
        x = x.reshape(-1, self.mean.numel()).float()
        n = x.shape[0]
        old_count = float(self.count)
        total = old_count + n
        mean, var = x.mean(0), x.var(0, unbiased=False)
        delta = mean - self.mean
        self.var.copy_((self.var * old_count + var * n + delta.square() * old_count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.fill_(total)

    def std(self) -> torch.Tensor:
        return (self.var + self.eps ** 2).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()


# ==============================================================================
# 4. SuperMimic WorldModel (适配 Ecow 四足机器人)
# ==============================================================================
class WorldModel(nn.Module):
    def __init__(
        self,
        joints: int = 12,
        frame_dim: int = 45,
        hidden: tuple[int, ...] = (2048, 2048, 2048),
        contact_dim: int = 4,
        default_joint_pos: torch.Tensor | None = None,
        joint_limits: torch.Tensor | None = None,
        nominal_height: float = 0.68,
        min_height: float = 0.32,
        max_tilt_deg: float = 65.0,
        dt: float = 0.02,
    ):
        super().__init__()
        self.joints = joints
        self.frame_dim = frame_dim
        self.contact_dim = contact_dim
        self.hidden = tuple(hidden)
        self.state_dim = 13 + 2 * joints
        self.increment_dim = 12 + 2 * joints
        self.nominal_height = nominal_height
        self.min_height = min_height
        self.cos_max_tilt = math.cos(math.radians(max_tilt_deg))
        self.dt = dt

        # 输入维度: state_features(13 + 2J) + action(J)
        self.inp_dim = (13 + 2 * joints) + joints
        layers = []
        width = self.inp_dim
        for h in hidden:
            layers.extend((nn.Linear(width, h), nn.LayerNorm(h), nn.SiLU()))
            width = h
        self.trunk = nn.Sequential(*layers)
        self.dynamics = nn.Linear(width, self.increment_dim)
        self.contact = nn.Linear(width, contact_dim)
        # 任务头预测: [reward(1), death_logit(1)]
        self.task_head = nn.Linear(width, 2)

        # 关键初始化：动力学增量头与任务头置零，保证初始多步滚动稳定
        nn.init.zeros_(self.dynamics.weight)
        nn.init.zeros_(self.dynamics.bias)
        nn.init.zeros_(self.task_head.weight)
        nn.init.zeros_(self.task_head.bias)

        self.input_norm = RunningNorm(self.inp_dim, eps=1e-2)
        self.output_norm = RunningNorm(self.increment_dim, eps=1e-4)
        self.reward_norm = RunningNorm(1, eps=1e-2)

        if default_joint_pos is not None:
            self.register_buffer("q0", default_joint_pos.clone().detach().view(-1))
        else:
            self.register_buffer("q0", torch.zeros(joints))

        if joint_limits is not None:
            self.register_buffer("limits", joint_limits.clone().detach().view(joints, 2))
        else:
            self.register_buffer("limits", torch.stack((-torch.ones(joints) * 3.14, torch.ones(joints) * 3.14), dim=-1))

    def inputs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat((state_features(state, self.joints), action), dim=-1)

    @torch.no_grad()
    def update_stats(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ):
        if valid_mask is not None and valid_mask.any():
            state = state[valid_mask]
            next_state = next_state[valid_mask]
            action = action[valid_mask]
            if reward is not None:
                reward = reward[valid_mask]
        self.input_norm.update(self.inputs(state, action))
        self.output_norm.update(state_delta(state, next_state, self.joints))
        if reward is not None and reward.numel() > 0:
            self.reward_norm.update(reward.unsqueeze(-1))

    def error_scale(self) -> torch.Tensor:
        floor = self.output_norm.mean.new_tensor(
            [0.003] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3
            + [0.003] * self.joints + [0.05] * self.joints
        )
        return torch.maximum(self.output_norm.std(), floor)

    def predict(self, state: torch.Tensor, action: torch.Tensor):
        x = self.input_norm(self.inputs(state, action))
        hidden = self.trunk(x)
        inc_norm = self.dynamics(hidden)
        contact_logits = self.contact(hidden)
        task_out = self.task_head(hidden)

        increments = inc_norm.float() * self.output_norm.std() + self.output_norm.mean
        next_state = state_integrate(state, increments, self.joints)
        pred_reward = task_out[..., 0].float() * self.reward_norm.std()[0] + self.reward_norm.mean[0]
        death_logit = task_out[..., 1].float()
        return next_state, contact_logits.float(), pred_reward, death_logit

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.predict(state, action)[:2]

    def build_actor_frame(
        self,
        next_state: torch.Tensor,
        action: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """从预测物理状态解析构建 Ecow 的 45 维单帧观测 (完全可微)
        严格对应 ecow/velocity_env_cfg.py 中 PolicyCfg 的顺序与 scale:
        1. velocity_commands (3, scale=1.0)
        2. base_ang_vel (3, scale=0.25)
        3. projected_gravity (3, scale=1.0)
        4. joint_pos_rel (12, scale=1.0)
        5. joint_vel_rel (12, scale=0.05)
        6. last_action (12, scale=1.0)
        """
        pos, quat, _, ang_vel_w, q, dq = unpack_state(next_state, self.joints)
        R = quat_to_mat(quat)
        Rt = R.transpose(-1, -2)

        base_ang_vel_b = mv(Rt, ang_vel_w) * 0.25
        gravity_w = torch.zeros_like(pos)
        gravity_w[..., 2] = -1.0
        projected_gravity_b = mv(Rt, gravity_w)
        joint_pos_rel = q - self.q0
        joint_vel_rel = dq * 0.05

        frame = torch.cat(
            (command, base_ang_vel_b, projected_gravity_b, joint_pos_rel, joint_vel_rel, action),
            dim=-1,
        )
        return torch.clamp(frame, -100.0, 100.0)

    def build_amp_obs(self, state: torch.Tensor) -> torch.Tensor:
        """从物理状态构建可微 AMP 观测特征，供判别器在 BPTT 滚动中计算风格奖励梯度"""
        pos, quat, lin_vel_w, ang_vel_w, q, dq = unpack_state(state, self.joints)
        R = quat_to_mat(quat)
        Rt = R.transpose(-1, -2)
        lin_vel_b = mv(Rt, lin_vel_w)
        ang_vel_b = mv(Rt, ang_vel_w)
        gravity_w = torch.zeros_like(pos)
        gravity_w[..., 2] = -1.0
        proj_grav_b = mv(Rt, gravity_w)
        return torch.cat((q, dq, lin_vel_b, ang_vel_b, proj_grav_b, pos[..., 2:3]), dim=-1)

    def compute_differentiable_reward_and_continuation(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        pre_pre_action: torch.Tensor,
        command: torch.Tensor,
        pred_reward: torch.Tensor,
        death_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对齐 supermimic/mdp/velocity.py, recovery.py 与 ecow/velocity_env_cfg.py 的可微奖励与软存活函数"""
        _, _, _, _, _, old_dq = unpack_state(state, self.joints)
        pos, quat, lin_vel_w, ang_vel_w, q, dq = unpack_state(next_state, self.joints)
        R = quat_to_mat(quat)
        Rt = R.transpose(-1, -2)
        R_yaw = yaw_mat(R)

        # 1. 线速度与角速度跟踪 (对齐 ecow/velocity_env_cfg.py: weight=2.0/1.0, std^2=0.1)
        lin_vel_local = mv(R_yaw.transpose(-1, -2), lin_vel_w)
        lin_err = (lin_vel_local[..., :2] - command[..., :2]).square().sum(-1)
        rew_lin_vel = 2.0 * torch.exp(-lin_err / 0.1)

        ang_vel_b = mv(Rt, ang_vel_w)
        ang_err = (ang_vel_b[..., 2] - command[..., 2]).square()
        rew_ang_vel = 1.0 * torch.exp(-ang_err / 0.1)

        # 2. 机身姿态、垂直速度与横滚俯仰角速度惩罚
        gravity_w = torch.zeros_like(pos)
        gravity_w[..., 2] = -1.0
        proj_grav_b = mv(Rt, gravity_w)
        up_z = -proj_grav_b[..., 2]

        pen_lin_vel_z = -1.0 * lin_vel_local[..., 2].square()
        pen_ang_vel_xy = -0.25 * ang_vel_b[..., :2].square().sum(-1)
        pen_flat_ori = -0.5 * proj_grav_b[..., :2].square().sum(-1)
        pen_adapt_ori = -0.3 * (proj_grav_b[..., 1].square() + 0.2 * proj_grav_b[..., 0].square())

        # 3. 机身标称高度保持 (引导 ecow 稳定站立，防止趴地)
        rew_height = 0.5 * torch.exp(-(pos[..., 2] - self.nominal_height).square() / (0.08 ** 2))

        # 4. 动作平滑度、二阶差分、关节加速度、关节速度与软限位惩罚 (防止 BPTT 梯度攻击的核心正则项)
        pen_action_rate = -0.01 * (action - prev_action).square().sum(-1)
        pen_smoothness = -0.01 * (action - 2.0 * prev_action + pre_pre_action).square().sum(-1)
        acc = (dq - old_dq) / self.dt
        pen_joint_acc = -2.5e-7 * acc.square().sum(-1)
        pen_joint_vel = -1e-4 * dq.square().sum(-1)
        pen_limits = -5.0 * (
            (self.limits[:, 0] - q).clamp_min(0.0) + (q - self.limits[:, 1]).clamp_min(0.0)
        ).sum(-1)

        # 5. 关节默认姿态正则 (对应 joint_pos 与 joint_pos_hip)
        cmd_norm = torch.linalg.norm(command, dim=-1)
        alpha = torch.clamp(cmd_norm / 0.2, min=0.0, max=1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        q_diff_norm = torch.linalg.norm(q - self.q0 + 1e-6, dim=-1)
        pen_joint_pos = -0.02 * ((1.0 - alpha) * 10.0 * q_diff_norm + alpha * q_diff_norm)

        # 6. 可微软存活率 (Soft Continuation) 与终止惩罚
        analytic_alive = (
            torch.sigmoid((pos[..., 2] - self.min_height) / 0.02)
            * torch.sigmoid((up_z - self.cos_max_tilt) / 0.03)
        )
        learned_alive = torch.sigmoid(-death_logit)
        continuation = 0.7 * analytic_alive + 0.3 * learned_alive
        death = 1.0 - continuation
        pen_termination = -200.0 * death

        analytic_step_reward = self.dt * (
            rew_lin_vel
            + rew_ang_vel
            + rew_height
            + pen_lin_vel_z
            + pen_ang_vel_xy
            + pen_flat_ori
            + pen_adapt_ori
            + pen_action_rate
            + pen_smoothness
            + pen_joint_acc
            + pen_joint_vel
            + pen_limits
            + pen_joint_pos
            + pen_termination
        )
        # 80% 精确解析可微奖励 + 20% 学习的环境奖励头（兼顾解析梯度精度与触地/抬腿等复杂环境项）
        total_reward = 0.8 * analytic_step_reward + 0.2 * pred_reward
        return total_reward, continuation


def compute_wm_rollout_loss(model: WorldModel, window: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """8 步自回归世界模型损失函数 (对齐 supermimic/core/world_model.py 与 task_model.py)"""
    state = window["state"][:, 0]
    mask = window.get("_mask", torch.ones_like(window["terminated"], dtype=torch.bool))
    horizon = window["action"].shape[1]
    scale = model.error_scale()

    physics_losses, contact_losses, aux_losses = [], [], []

    for t in range(horizon):
        state = torch.where(mask[:, t, None], state, window["state"][:, 0])
        state, contact_logits, pred_rew, death_logit = model.predict(state, window["action"][:, t])
        w = mask[:, t].float()
        denom = w.sum().clamp_min(1.0)

        # 1. 局部状态增量 MSE
        err = state_delta(window["next_state"][:, t], state, model.joints) / scale
        physics_losses.append((err.square().mean(-1) * w).sum() / denom)

        # 2. 足端接触 BCE
        if "feet_contact" in window:
            c_loss = F.binary_cross_entropy_with_logits(
                contact_logits, window["feet_contact"][:, t].float(), reduction="none"
            ).mean(-1)
            contact_losses.append((c_loss * w).sum() / denom)

        # 3. 奖励 MSE 与终止 BCE
        rew_err = ((pred_rew - window["reward"][:, t]) / model.reward_norm.std()[0]).square()
        death_loss = F.binary_cross_entropy_with_logits(
            death_logit, window["terminated"][:, t].float(), reduction="none"
        )
        aux_losses.append(((rew_err + death_loss) * w).sum() / denom)

    p_loss = torch.stack(physics_losses).mean()
    c_loss = torch.stack(contact_losses).mean() if contact_losses else torch.zeros_like(p_loss)
    a_loss = torch.stack(aux_losses).mean()
    total_loss = p_loss + 0.1 * c_loss + 0.5 * a_loss

    return total_loss, {
        "wm_loss": total_loss.detach(),
        "wm_physics_loss": p_loss.detach(),
        "wm_contact_loss": c_loss.detach(),
        "wm_aux_loss": a_loss.detach(),
    }


# ==============================================================================
# 5. BPTT 专用 Critic (ValueModel) 与 GAE 目标计算 (对齐 supermimic/core/imagination.py)
# ==============================================================================
class BPTTValueModel(nn.Module):
    def __init__(self, wm_inp_dim: int, command_dim: int = 3):
        super().__init__()
        in_dim = wm_inp_dim + command_dim
        self.norm = RunningNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)


def critic_input(wm: WorldModel, state: torch.Tensor, prev_action: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    base = wm.input_norm(wm.inputs(state, prev_action))
    return torch.cat((base, command), dim=-1)


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor,
    continuations: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    result = values[-1]
    targets = []
    for t in reversed(range(rewards.shape[0])):
        result = rewards[t] + gamma * continuations[t] * ((1.0 - lam) * values[t + 1] + lam * result)
        targets.append(result)
    return torch.stack(targets[::-1], dim=0)


def update_bptt_critic(
    critic: BPTTValueModel,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    epochs: int = 5,
    minibatches: int = 4,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    x = rollout["critic_inputs"].detach()
    critic.norm.update(x)
    with torch.no_grad():
        values = critic(x)
        targets = gae_targets(
            rollout["rewards"].detach(),
            rollout["continuations"].detach(),
            values,
            gamma=gamma,
            lam=lam,
        )
    x_flat = x[:-1].flatten(0, 1)
    targets_flat = targets.flatten()
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(x_flat), device=x_flat.device).tensor_split(minibatches):
            if not len(indices):
                continue
            loss = (critic(x_flat[indices]) - targets_flat[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
    return torch.stack(losses).mean()
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class JointDataCollector:
    def __init__(self, env, enabled=True):
        """
            env:包装好的HimlocoVecEnvWrapper
            enabled:False 都是no-op
        """
        self._enabled = enabled
        if not enabled:
            return
        self._robot = env.unwrapped.scene["robot"]
        self._cmd_manager = env.unwrapped.command_manager  # 命令管理器
        self._action_manager = env.unwrapped.action_manager
        self._joint_names = self._robot.data.joint_names   # 12 个关节名
        self._num_dof = self._robot.num_dof
        self._leg_groups = self._build_leg_groups()

        self.cmd_vel_history = []
        self.actual_vel_history = []
        self.cmd_joint_pos_history = []
        self.actual_joint_pos_history = []
        self.joint_vel_history = []
        self.joint_torque_history = []

    def _build_leg_groups(self):
        """按腿前缀（fr_, fl_, rl_, rr_）对关节索引分组"""
        groups = {}
        for i, name in enumerate(self._joint_names):
            prefix = name.split("_")[0]
            groups.setdefault(prefix, []).append(i)
        return groups

    def step(self):
        if not self._enabled:
            return
        # 速度指令
        cmd = self._cmd_manager.get_command("base_velocity")
        self.cmd_vel_history.append(cmd[0].cpu().numpy())

        # 实际速度
        lin_vel = self._robot.data.root_lin_vel_b[0]
        ang_vel = self._robot.data.root_ang_vel_b[0]
        self.actual_vel_history.append(torch.cat([lin_vel[:2], ang_vel[2:3]]).cpu().numpy())

        # 关节位置指令
        action = self._action_manager.action[0]
        self.cmd_joint_pos_history.append(action.cpu().numpy())

        # 实际关节数据
        self.actual_joint_pos_history.append(
            self._robot.data.joint_pos[0].cpu().numpy()
        )
        self.joint_vel_history.append(
            self._robot.data.joint_vel[0].cpu().numpy()
        )
        self.joint_torque_history.append(
            self._robot.data.applied_torque[0].cpu().numpy()
        )

    # ------------------------------------------------------------------
    # 绘图入口
    # ------------------------------------------------------------------
    def plot(self, save_dir=None):
        """显示图表。如果指定 save_dir 则同时保存到该目录。"""
        if not self._enabled:
            return
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self._plot_velocity_tracking(save_dir)
        self._plot_joint_tracking(save_dir)
        self._plot_joint_vel_torque(save_dir)
        plt.show()

    # ------------------------------------------------------------------
    # 图1：速度追踪  (v_x, v_y, ω_z)  —  指令 vs 实际
    # ------------------------------------------------------------------
    def _plot_velocity_tracking(self, save_dir):
        if not self.cmd_vel_history:
            return

        cmd = np.stack(self.cmd_vel_history, axis=0)        # (T, 3)
        act = np.stack(self.actual_vel_history, axis=0)     # (T, 3)
        t = np.arange(len(cmd))

        labels = ["v_x (m/s)", "v_y (m/s)", "ω_z (rad/s)"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for i, ax in enumerate(axes):
            ax.plot(t, cmd[:, i], linewidth=1.5, label="cmd", color="dodgerblue")
            ax.plot(t, act[:, i], linewidth=1.5, label="actual", color="tomato", linestyle="--")
            ax.set_title(labels[i])
            ax.set_xlabel("step")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Base Velocity Tracking", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "velocity_tracking.png"), dpi=150)

    # ------------------------------------------------------------------
    # 图2：关节位置追踪  —  指令（policy action） vs 实际（joint_pos）
    #      按腿分 4 个子图（FR / FL / RL / RR）
    # ------------------------------------------------------------------
    def _plot_joint_tracking(self, save_dir):
        if not self.cmd_joint_pos_history:
            return

        cmd = np.stack(self.cmd_joint_pos_history, axis=0)       # (T, 12)
        act = np.stack(self.actual_joint_pos_history, axis=0)    # (T, 12)
        t = np.arange(len(cmd))

        # 按 FR / FL / RL / RR 顺序排列子图
        leg_order = ["fr", "fl", "rl", "rr"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for ax_idx, leg_prefix in enumerate(leg_order):
            ax = axes[ax_idx]
            if leg_prefix not in self._leg_groups:
                continue
            indices = self._leg_groups[leg_prefix]
            for j in indices:
                name = self._joint_names[j]
                # 简化标签：去掉 _joint 后缀
                short_name = name.replace("_joint", "")
                ax.plot(t, cmd[:, j], linewidth=1.0, alpha=0.8,
                        label=f"cmd {short_name}")
                ax.plot(t, act[:, j], linewidth=1.0, alpha=0.8,
                        label=f"act {short_name}", linestyle="--")

            ax.set_title(f"{leg_prefix.upper()} Leg")
            ax.set_xlabel("step")
            ax.set_ylabel("joint position (rad)")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Joint Position Tracking (cmd vs actual)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "joint_position_tracking.png"), dpi=150)

    # ------------------------------------------------------------------
    # 图3：关节速度 & 扭矩
    # ------------------------------------------------------------------
    def _plot_joint_vel_torque(self, save_dir):
        if not self.joint_vel_history:
            return

        vel = np.stack(self.joint_vel_history, axis=0)         # (T, 12)
        trq = np.stack(self.joint_torque_history, axis=0)      # (T, 12)
        t = np.arange(len(vel))

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))

        # 上图：关节速度
        ax = axes[0]
        for j in range(self._num_dof):
            short_name = self._joint_names[j].replace("_joint", "")
            ax.plot(t, vel[:, j], linewidth=1.0, alpha=0.8, label=short_name)
        ax.set_title("Joint Velocities")
        ax.set_ylabel("rad/s")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

        # 下图：关节扭矩
        ax = axes[1]
        for j in range(self._num_dof):
            short_name = self._joint_names[j].replace("_joint", "")
            ax.plot(t, trq[:, j], linewidth=1.0, alpha=0.8, label=short_name)
        ax.set_title("Joint Torques")
        ax.set_xlabel("step")
        ax.set_ylabel("N·m")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Joint Velocities & Torques", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "joint_vel_torque.png"), dpi=150)

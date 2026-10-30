from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils
import numpy as np
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformLevelVelocityCommandCfg



class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: UniformLevelVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_straight_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_rotation_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_lateral_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_high_vel_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env = env

        # 判断是否包含地形模块
        self.has_terrain = hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_types")
        
        if self.has_terrain:
            terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
            sub_terrains_cfg = terrain_gen_cfg.sub_terrains
            sub_terrain_names = list(sub_terrains_cfg.keys())

            # terrain_types 的真实取值上界 = 网格列数
            num_terrain_types = int(terrain_gen_cfg.num_cols)

            # ---------- 1. 按 proportion 计算每一列 (terrain_type) 对应的 sub_terrain ----------
            proportions = np.array(
                [sub_terrains_cfg[name].proportion for name in sub_terrain_names],
                dtype=np.float64,
            )
            proportions = proportions / proportions.sum()          # 归一化，保证总和为 1
            target_cols = proportions * num_terrain_types          # 期望列数（浮点）

            # 整数化：先向下取整，再把剩余的列补给“小数部分最大”的 sub_terrain
            num_cols_per_terrain = np.floor(target_cols).astype(int)
            remainder = target_cols - num_cols_per_terrain
            remaining = num_terrain_types - num_cols_per_terrain.sum()
            if remaining > 0:
                top_idx = np.argsort(-remainder)[:remaining]
                num_cols_per_terrain[top_idx] += 1

            # 构建 terrain_type -> sub_terrain_name 的映射表（长度 = num_terrain_types）
            terrain_type_to_name: list[str] = []
            for name, n in zip(sub_terrain_names, num_cols_per_terrain):
                terrain_type_to_name.extend([name] * int(n))

            # 保险：万一个别版本分配策略略有差异，长度不足时用最后一种补全
            while len(terrain_type_to_name) < num_terrain_types:
                terrain_type_to_name.append(sub_terrain_names[-1])

            # ---------- 2. 用真实上界初始化张量 ----------
            self.terrain_vel_limits  = torch.zeros((num_terrain_types, 2), device=self.device)
            self.terrain_vel_current = torch.zeros((num_terrain_types, 2), device=self.device)

            for terrain_type, terrain_name in enumerate(terrain_type_to_name):
                limit = self.cfg.terrain_limit_ranges.get(
                    terrain_name, self.cfg.curriculums_limit_ranges
                )
                self.terrain_vel_limits[terrain_type, 0] = limit[0]
                self.terrain_vel_limits[terrain_type, 1] = limit[1]

                # 当前等级从基础范围开始
                self.terrain_vel_current[terrain_type, 0] = self.cfg.ranges.lin_vel_x[0]
                self.terrain_vel_current[terrain_type, 1] = self.cfg.ranges.lin_vel_x[1]

                print(
                    f"[terrain_type {terrain_type:2d}] {terrain_name:>22s} "
                    f"limit = [{limit[0]:.2f}, {limit[1]:.2f}]"
                )

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        # sample velocity commands
        r = torch.empty(len(env_ids_tensor), device=self.device)
        self.vel_command_b[env_ids_tensor, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        self.vel_command_b[env_ids_tensor, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids_tensor, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        
        if self.cfg.heading_command:
            self.heading_target[env_ids_tensor] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # high_vel_threshold = int(self.num_envs * self.cfg.rel_high_vel_envs)
        # high_vel_mask = (env_ids_tensor <= high_vel_threshold)
        # high_vel_env_ids = env_ids_tensor[high_vel_mask]
        self.is_high_vel_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_high_vel_envs
        high_vel_mask = self.is_high_vel_env[env_ids_tensor]
        high_vel_env_ids = env_ids_tensor[high_vel_mask]
        
        # 核心修改：利用张量切片精准映射高速组环境的速度范围
        if self.has_terrain:
            env_terrain_types = self.env.scene.terrain.terrain_types[high_vel_env_ids].long()
            current_ranges = self.terrain_vel_current[env_terrain_types]
            
            # 使用公式 v = min + r * (max - min) 进行对应地形的张量化均匀采样
            r_high = torch.rand(len(high_vel_env_ids), device=self.device)
            sampled_vels = current_ranges[:, 0] + r_high * (current_ranges[:, 1] - current_ranges[:, 0])
            self.vel_command_b[high_vel_env_ids, 0] = sampled_vels
        else:
            r_high = torch.empty(len(high_vel_env_ids), device=self.device)
            self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)

        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & \
                            (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range

        if self.cfg.straight_target_prob > 0.0:
            self.is_straight_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.straight_target_prob
        
        if self.cfg.rotation_target_prob > 0.0:
            self.is_rotation_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rotation_target_prob
        
        if self.cfg.lateral_target_prob > 0.0:
            self.is_lateral_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.lateral_target_prob
        
        if self.cfg.rel_standing_envs > 0.0:
            self.is_standing_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # set small commands to zero
        self.vel_command_b[env_ids_tensor, :2] *= (torch.norm(self.vel_command_b[env_ids_tensor, :2], dim=1) > \
                                            self.cfg.min_command_norm).unsqueeze(1)
        self.vel_command_b[env_ids_tensor, 2] *= (torch.abs(self.vel_command_b[env_ids_tensor, 2]) > \
                                            self.cfg.min_command_norm)
        
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            self.vel_command_b[env_ids, 2] *= (torch.abs(self.vel_command_b[env_ids, 2]) > \
                                            self.cfg.min_command_norm)
        
        lateral_env_ids = self.is_lateral_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[lateral_env_ids, 0] = 0.0
        self.vel_command_b[lateral_env_ids, 2] = 0.0

        rotation_env_ids = self.is_rotation_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[rotation_env_ids, 0:2] = 0.0
        
        straight_env_ids = self.is_straight_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[straight_env_ids, 1:3] = 0.0
        
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
    """Debug Visualization"""

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     if debug_vis:
    #         if not hasattr(self, "goal_vel_visualizer"):
    #             self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
    #             self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
    #         self.goal_vel_visualizer.set_visibility(True)
    #         self.current_vel_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_vel_visualizer"):
    #             self.goal_vel_visualizer.set_visibility(False)
    #             self.current_vel_visualizer.set_visibility(False)
    
    # goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/velocity_goal"
    # )
    # """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/velocity_current"
    # )
    # """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # def _debug_vis_callback(self, event):
    #     if not self.robot.is_initialized:
    #         return
    #     base_pos_w = self.robot.data.root_pos_w.clone()
    #     base_pos_w[:, 2] += 0.5
    #     vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
    #     vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
    #     self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
    #     self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    # def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Converts the XY base velocity command to arrow direction rotation."""
    #     default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
    #     arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
    #     arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
    #     heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    #     zeros = torch.zeros_like(heading_angle)
    #     arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
    #     base_quat_w = self.robot.data.root_quat_w
    #     arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

    #     return arrow_scale, arrow_quat
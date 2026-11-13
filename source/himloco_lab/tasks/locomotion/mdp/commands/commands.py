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
from ..utils import sample_disjoint_intervals



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
            num_terrain_types = int(terrain_gen_cfg.num_cols)

            # 严格对齐 Isaac Lab TerrainGenerator 的 np.cumsum 列分配规则
            proportions = np.array(
                [sub_terrains_cfg[name].proportion for name in sub_terrain_names],
                dtype=np.float64,
            )
            proportions = proportions / proportions.sum()
            cumsum_props = np.cumsum(proportions)

            terrain_type_to_name: list[str] = []
            for col_idx in range(num_terrain_types):
                sub_idx = int(np.min(np.where(col_idx / num_terrain_types + 0.001 < cumsum_props)[0]))
                terrain_type_to_name.append(sub_terrain_names[sub_idx])

            self.terrain_vel_limits = torch.zeros((num_terrain_types, 2), device=self.device)
            self.terrain_vel_current = torch.zeros((num_terrain_types, 2), device=self.device)

            for terrain_type, terrain_name in enumerate(terrain_type_to_name):
                limit = self.cfg.terrain_limit_ranges.get(
                    terrain_name, self.cfg.curriculums_limit_ranges
                )
                self.terrain_vel_limits[terrain_type, 0] = limit[0]
                self.terrain_vel_limits[terrain_type, 1] = limit[1]
                self.terrain_vel_current[terrain_type, 0] = self.cfg.ranges.lin_vel_x[0]
                self.terrain_vel_current[terrain_type, 1] = self.cfg.ranges.lin_vel_x[1]

                print(
                    f"[terrain_type {terrain_type:2d}] {terrain_name:>22s} "
                    f"limit = [{limit[0]:.2f}, {limit[1]:.2f}]"
                )

        # 记录单局内机器人的历史最大移动距离 & 速度指令矢量累加
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        # 新增：记录每个环境最近完整回合的速度追踪奖励率 EMA（供课程升级稳定评估）
        self.episode_track_rew_ema = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_completed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 计算有效地形边长（用于动态重采样下界计算）
        terrain_cfg = self._env.scene.terrain.cfg
        if terrain_cfg.terrain_type == "generator" and terrain_cfg.terrain_generator is not None:
            gen_cfg = terrain_cfg.terrain_generator
            border = getattr(gen_cfg, "sub_terrain_border_width", 0.0) or 0.0
            self.terrain_length = max(0.0, gen_cfg.size[0] - 2.0 * border)
        else:
            self.terrain_length = 0.0

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        #在环境 reset 时清零过程最大距离与累积指令
        self.max_move_distance[env_ids] = 0.0
        self.commands_xy_accumulation[env_ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        if len(env_ids_tensor) == 0:
            return

        r = torch.empty(len(env_ids_tensor), device=self.device)
        self.vel_command_b[env_ids_tensor, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        self.vel_command_b[env_ids_tensor, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids_tensor, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids_tensor] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_high_vel_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_high_vel_envs
        high_vel_mask = self.is_high_vel_env[env_ids_tensor]
        high_vel_env_ids = env_ids_tensor[high_vel_mask]

        # 为每个重采样环境保存其对应的 vx 课程上下限
        env_cfg_min = torch.full((len(env_ids_tensor),), self.cfg.low_vel_env_lin_x_ranges[0], device=self.device)
        env_cfg_max = torch.full((len(env_ids_tensor),), self.cfg.low_vel_env_lin_x_ranges[1], device=self.device)

        if len(high_vel_env_ids) > 0:
            if self.has_terrain:
                env_terrain_types = self.env.scene.terrain.terrain_types[high_vel_env_ids].long()
                current_ranges = self.terrain_vel_current[env_terrain_types]
                r_high = torch.rand(len(high_vel_env_ids), device=self.device)
                sampled_vels = current_ranges[:, 0] + r_high * (current_ranges[:, 1] - current_ranges[:, 0])
                self.vel_command_b[high_vel_env_ids, 0] = sampled_vels
                env_cfg_min[high_vel_mask] = current_ranges[:, 0]
                env_cfg_max[high_vel_mask] = current_ranges[:, 1]
            else:
                r_high = torch.empty(len(high_vel_env_ids), device=self.device)
                self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)
                env_cfg_min[high_vel_mask] = self.cfg.ranges.lin_vel_x[0]
                env_cfg_max[high_vel_mask] = self.cfg.ranges.lin_vel_x[1]

        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & (
            self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max
        )
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range

        if self.cfg.straight_target_prob > 0.0:
            self.is_straight_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.straight_target_prob
        if self.cfg.rotation_target_prob > 0.0:
            self.is_rotation_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rotation_target_prob
        if self.cfg.lateral_target_prob > 0.0:
            self.is_lateral_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.lateral_target_prob
        if self.cfg.rel_standing_envs > 0.0:
            self.is_standing_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # 立即应用姿态模式置零，防止后续 dynamic_resample 和 commands_xy_accumulation 累加虚假指令
        lat_mask = self.is_lateral_env[env_ids_tensor]
        rot_mask = self.is_rotation_env[env_ids_tensor]
        str_mask = self.is_straight_env[env_ids_tensor]
        std_mask = self.is_standing_env[env_ids_tensor]

        self.vel_command_b[env_ids_tensor[lat_mask], 0] = 0.0
        self.vel_command_b[env_ids_tensor[lat_mask], 2] = 0.0
        self.vel_command_b[env_ids_tensor[rot_mask], 0:2] = 0.0
        self.vel_command_b[env_ids_tensor[str_mask], 1:3] = 0.0
        self.vel_command_b[env_ids_tensor[std_mask], :] = 0.0

        # 死区过滤
        self.vel_command_b[env_ids_tensor, :2] *= (
            torch.norm(self.vel_command_b[env_ids_tensor, :2], dim=1) > self.cfg.min_command_norm
        ).unsqueeze(1)
        self.vel_command_b[env_ids_tensor, 2] *= (
            torch.abs(self.vel_command_b[env_ids_tensor, 2]) > self.cfg.min_command_norm
        )

        # 修复后的动态下界采样：仅对非静止、非纯旋转、非纯侧移的【高速组】环境使用正确的 env_cfg_min/max
        if getattr(self.cfg, "dynamic_resample_commands", False) and self.terrain_length > 0.0:
            mean_resample_dt = 0.5 * (self.cfg.resampling_time_range[0] + self.cfg.resampling_time_range[1])
            remaining_dist = torch.clip(
                0.5 * self.terrain_length
                - torch.norm(self.commands_xy_accumulation[env_ids_tensor], dim=1) * mean_resample_dt,
                min=0.0,
            )
            rem_steps = (self._env.max_episode_length - self._env.episode_length_buf[env_ids_tensor] + 1).clamp(min=1)
            vel_low_bound = torch.clip(remaining_dist / (rem_steps * self._env.step_dt + 1e-9), min=0.0)

            active_vx_mask = (
                high_vel_mask
                & (~std_mask)
                & (~rot_mask)
                & (~lat_mask)
                & (torch.abs(self.vel_command_b[env_ids_tensor, 0]) > 1e-3)
            )
            active_ids = env_ids_tensor[active_vx_mask]
            if len(active_ids) > 0:
                cfg_min = env_cfg_min[active_vx_mask]
                cfg_max = env_cfg_max[active_vx_mask]
                bound = torch.minimum(vel_low_bound[active_vx_mask], 0.6 * torch.maximum(cfg_max.abs(), cfg_min.abs()))
                self.vel_command_b[active_ids, 0] = sample_disjoint_intervals(
                    active_ids, bound, cfg_min, cfg_max, self.device
                )
                # 速度超过低速区间时清零侧向速度 vy，防止斜坡高速大侧移摔倒
                high_vx = (self.vel_command_b[active_ids, 0] < low_vel_x_min) | (
                    self.vel_command_b[active_ids, 0] > low_vel_x_max
                )
                self.vel_command_b[active_ids[high_vx], 1] = 0.0

        self.commands_xy_accumulation[env_ids_tensor] += self.vel_command_b[env_ids_tensor, :2]
        
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

        # 在每个仿真步实时更新 Episode 内的历史最大移动半径
        current_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2], dim=1)
        self.max_move_distance = torch.maximum(self.max_move_distance, current_dist)
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
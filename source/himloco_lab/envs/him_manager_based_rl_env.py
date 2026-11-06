# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

def _remove_online_usd_dependencies(cfg: ManagerBasedRLEnvCfg) -> None:
    """移除环境配置中所有对 AWS S3 在线 USD / MDL / HDR 资源的依赖，确保 100% 离线可运行。"""
    # 1. 处理 commands 中的在线 arrow_x.usd 依赖
    if hasattr(cfg, "commands") and cfg.commands is not None:
        for cmd_name in dir(cfg.commands):
            if cmd_name.startswith("_"):
                continue
            cmd_term = getattr(cfg.commands, cmd_name, None)
            if cmd_term is None:
                continue
            # 关闭训练期不必要的命令箭头可视化，并将默认在线 arrow_x.usd 替换为本地程序化几何体
            if hasattr(cmd_term, "debug_vis"):
                cmd_term.debug_vis = False
            if hasattr(cmd_term, "goal_vel_visualizer_cfg") and cmd_term.goal_vel_visualizer_cfg is not None:
                cmd_term.goal_vel_visualizer_cfg.markers = {
                    "arrow": sim_utils.CuboidCfg(
                        size=(0.3, 0.06, 0.06),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )
                }
            if hasattr(cmd_term, "current_vel_visualizer_cfg") and cmd_term.current_vel_visualizer_cfg is not None:
                cmd_term.current_vel_visualizer_cfg.markers = {
                    "arrow": sim_utils.CuboidCfg(
                        size=(0.3, 0.06, 0.06),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
                    )
                }

    # # 2. 处理 scene 中的在线地面材质 (.mdl)、天空盒 (.hdr) 与射线可视化
    # if hasattr(cfg, "scene") and cfg.scene is not None:
    #     # 替换在线地面 MDL 材质为本地 PreviewSurfaceCfg
    #     if hasattr(cfg.scene, "terrain") and cfg.scene.terrain is not None:
    #         vis_mat = getattr(cfg.scene.terrain, "visual_material", None)
    #         if vis_mat is not None and hasattr(vis_mat, "mdl_path"):
    #             mdl_path = str(getattr(vis_mat, "mdl_path", ""))
    #             if mdl_path.startswith("http") or "omniverse" in mdl_path or "Nucleus" in mdl_path:
    #                 cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
    #                     diffuse_color=(0.25, 0.25, 0.25)
    #                 )

    #     # 移除在线天空盒 HDR 贴图，改用纯色环境光
    #     if hasattr(cfg.scene, "sky_light") and cfg.scene.sky_light is not None:
    #         spawn_cfg = getattr(cfg.scene.sky_light, "spawn", None)
    #         if spawn_cfg is not None and hasattr(spawn_cfg, "texture_file"):
    #             tex_file = getattr(spawn_cfg, "texture_file", None)
    #             if tex_file is not None and (str(tex_file).startswith("http") or "omniverse" in str(tex_file)):
    #                 spawn_cfg.texture_file = None
    #                 spawn_cfg.color = (0.75, 0.75, 0.75)

    #     # 关闭所有 RayCaster 传感器的 debug_vis，避免创建大量可视化图元及潜在在线依赖
    #     for attr_name in dir(cfg.scene):
    #         if attr_name.startswith("_"):
    #             continue
    #         attr_val = getattr(cfg.scene, attr_name, None)
    #         if hasattr(attr_val, "debug_vis") and hasattr(attr_val, "ray_alignment"):
    #             attr_val.debug_vis = False

class HimlocoManagerBasedRLEnv(ManagerBasedRLEnv):
    """HimLoco environment with Manager-Based RL interface supporting both PPO and SuperMimic BPTT."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize the HimLoco environment."""
        # 在初始化底层环境与 Manager 之前，清除所有在线 USD/MDL/HDR 依赖
        _remove_online_usd_dependencies(cfg)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._pre_pre_action = torch.zeros(
            (self.num_envs, self.action_manager.total_action_dim), device=self.device
        )
        # AMP: cache foot body indices for get_amp_observations()
        robot = self.scene.articulations["robot"]
        self._amp_foot_body_indices, _ = robot.find_bodies(".*_foot")
        self.num_amp_obs = (
            robot.data.joint_pos.shape[1] * 2
            + len(self._amp_foot_body_indices) * 3
            + 3
            + 3
        )
        self._capturing_step: bool = False
        self.terminal_physics_snapshot: tuple[torch.Tensor, torch.Tensor] | None = None

    def _get_ground_height(self) -> torch.Tensor:
        """获取机身正下方的地形高度，使物理状态中的 root_pos_z 始终表示相对地面的真实离地高度。"""
        if "base_height_scanner" in self.scene.sensors:
            sensor = self.scene.sensors["base_height_scanner"]
            ray_hits_z = sensor.data.ray_hits_w[..., 2]
            valid = ~torch.isnan(ray_hits_z) & ~torch.isinf(ray_hits_z) & (torch.abs(ray_hits_z) < 1e4)
            if valid.any():
                masked_z = torch.where(valid, ray_hits_z, torch.zeros_like(ray_hits_z))
                count = valid.float().sum(dim=1).clamp_min(1.0)
                ground_z = masked_z.sum(dim=1) / count
                all_invalid = valid.sum(dim=1) == 0
                if all_invalid.any():
                    ground_z[all_invalid] = self.scene.env_origins[all_invalid, 2]
                return ground_z
        return self.scene.env_origins[:, 2]

    def read_physics_state(self) -> torch.Tensor:
        """读取 37 维物理状态 (对齐 SuperMimic WorldModel 格式):
        [root_pos_local(3), root_quat_wxyz(4), root_lin_vel_w(3),
         root_ang_vel_w(3), joint_pos(12), joint_vel(12)]
        """
        robot = self.scene.articulations["robot"]
        root_pos_xy = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        ground_z = self._get_ground_height()
        root_pos_z = (robot.data.root_pos_w[:, 2] - ground_z).unsqueeze(-1)
        root_pos = torch.cat((root_pos_xy, root_pos_z), dim=-1)

        root_quat_wxyz = robot.data.root_quat_w
        root_lin_vel_w = robot.data.root_lin_vel_w
        root_ang_vel_w = robot.data.root_ang_vel_w
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        return torch.cat(
            (root_pos, root_quat_wxyz, root_lin_vel_w, root_ang_vel_w, joint_pos, joint_vel),
            dim=-1,
        ).clone()

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Execute one time-step of the environment's dynamics and reset terminated environments."""
        # process actions
        self._pre_pre_action = self.action_manager.prev_action.clone()
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # -- get observations before reset
        self.obs_buf_before_reset = self.observation_manager.compute()
        # post-step:
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        self.terminal_physics_snapshot = None
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # -- capture terminal AMP states & 37D physics states BEFORE reset
            terminal_amp_states = self.get_amp_observations()[reset_env_ids].clone()
            self.terminal_physics_snapshot = (
                reset_env_ids.clone(),
                self.read_physics_state()[reset_env_ids].clone(),
            )

            self._reset_idx(reset_env_ids)
            self._pre_pre_action[reset_env_ids] = 0.0
            self.action_manager.reset(reset_env_ids)

            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)
        else:
            terminal_amp_states = self.get_amp_observations()[reset_env_ids]

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations after reset
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.obs_buf_before_reset,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
            reset_env_ids,
            self.obs_buf_before_reset.get("critic", self.obs_buf_before_reset["policy"]),
            terminal_amp_states,
        )

    @property
    def pre_pre_action(self) -> torch.Tensor:
        """The previous-previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._pre_pre_action

    @pre_pre_action.setter
    def pre_pre_action(self, value: torch.Tensor) -> None:
        self._pre_pre_action = value

    def get_amp_observations(self) -> torch.Tensor:
        """Get 42D AMP observations for discriminator training:
        [joint_pos(12), foot_pos_b(12), base_lin_vel_b(3), base_ang_vel_b(3), joint_vel(12)]
        """
        robot = self.scene.articulations["robot"]
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b

        footpos_translated = (
            robot.data.body_pos_w[:, self._amp_foot_body_indices, :]
            - robot.data.root_pos_w[:, :].unsqueeze(1)
        )
        footpos_in_body_frame = torch.zeros_like(footpos_translated)
        for i in range(len(self._amp_foot_body_indices)):
            footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
                robot.data.root_quat_w, footpos_translated[:, i, :]
            )
        foot_pos = footpos_in_body_frame.flatten(1)

        return torch.cat(
            [joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel], dim=-1
        )
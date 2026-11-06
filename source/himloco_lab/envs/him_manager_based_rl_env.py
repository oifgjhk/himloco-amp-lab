import torch
from typing import Dict, Any
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

class HimlocoManagerBasedRLEnv(ManagerBasedRLEnv):
    """HimLoco environment with Manager-Based RL interface."""
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize the HimLoco environment.
        
        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None.
            **kwargs: Additional keyword arguments (e.g., env_cfg_entry_point from gym.register).
        """
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._pre_pre_action = torch.zeros((self.num_envs, self.action_manager.total_action_dim), device=self.device)
        # AMP: cache foot body indices for get_amp_observations()
        robot = self.scene.articulations["robot"]
        self._amp_foot_body_indices, _ = robot.find_bodies(".*_foot")
        self.num_amp_obs = (robot.data.joint_pos.shape[1] * 2 +  # joint_pos + joint_vel
                            len(self._amp_foot_body_indices) * 3 +  # foot_pos
                            3 + 3 + 1)  # base_lin_vel + base_ang_vel + z_pos
        self._capturing_step = False
        self.terminal_physics_snapshot: tuple[torch.Tensor, torch.Tensor] | None = None
    
    def read_physics_state(self) -> torch.Tensor:
        """读取 37 维物理状态: [root_pos_local(3), root_quat_wxyz(4), root_lin_vel_w(3),
        root_ang_vel_w(3), joint_pos(12), joint_vel(12)]
        """
        robot = self.scene["robot"]
        root_pos = robot.data.root_pos_w - self.scene.env_origins
        root_quat_wxyz = robot.data.root_quat_w
        root_lin_vel_w = robot.data.root_lin_vel_w
        root_ang_vel_w = robot.data.root_ang_vel_w
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        return torch.cat(
            (root_pos, root_quat_wxyz, root_lin_vel_w, root_ang_vel_w, joint_pos, joint_vel),
            dim=-1,
        ).clone()

    def _reset_idx(self, env_ids: torch.Tensor):
        if self._capturing_step and len(env_ids) > 0:
            ids = env_ids.clone()
            self.terminal_physics_snapshot = (ids, self.read_physics_state()[ids].clone())
        super()._reset_idx(env_ids)
        if hasattr(self, "pre_pre_action") and len(env_ids) > 0:
            self.pre_pre_action[env_ids] = 0.0

    def step(self, action: torch.Tensor) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Capture observations before reset.
        4. Perform rendering if gui is enabled.
        5. Update the environment counters and compute the rewards and terminations.
        6. Reset the environments that terminated.
        7. Compute the observations after reset.
        8. Return the observations (after reset), observations (before reset), rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing:
                - obs_buf: Current observations after reset (Dict[str, torch.Tensor])
                - obs_buf_before_reset: Observations before reset (Dict[str, torch.Tensor])
                - reward_buf: Rewards for each environment (torch.Tensor)
                - reset_terminated: Termination flags (torch.Tensor)
                - reset_time_outs: Timeout flags (torch.Tensor)
                - extras: Additional information dictionary (Dict[str, Any])
        """
        # process actions
        self._pre_pre_action = self.action_manager.prev_action.clone()
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # -- get observations before reset
        self.obs_buf_before_reset = self.observation_manager.compute()
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
            

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            #self.extras['episode'] = dict()
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)
            
            

            # -- capture terminal AMP states before reset
            terminal_amp_states = self.get_amp_observations()[reset_env_ids]

            #reward_extras = self.reward_manager.reset(reset_env_ids)
            #self.extras['episode'].update(reward_extras)

            self._reset_idx(reset_env_ids)
            self._pre_pre_action[reset_env_ids] = 0.0
            self.action_manager.reset(reset_env_ids)
            
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)
        else:
            # -- capture terminal AMP states before reset
            terminal_amp_states = self.get_amp_observations()[reset_env_ids]
            # terminal_amp_states = torch.zeros(0, self.num_amp_obs, device=self.device)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return (self.obs_buf, self.obs_buf_before_reset, self.reward_buf,
                self.reset_terminated, self.reset_time_outs, self.extras,
                reset_env_ids, self.obs_buf_before_reset.get("critic", self.obs_buf_before_reset["policy"]),
                terminal_amp_states)
    @property
    def pre_pre_action(self) -> torch.Tensor:
        """The previous  previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._pre_pre_action
    
    # def get_amp_observations(self) -> torch.Tensor:
    #     """Get AMP observations for discriminator training.
        
    #     Returns tensor of shape (num_envs, num_amp_obs):
    #         joint_pos + foot_pos + base_lin_vel + base_ang_vel + joint_vel + z_pos
    #     """
    #     robot = self.scene.articulations["robot"]
    #     joint_pos = robot.data.joint_pos
    #     joint_vel = robot.data.joint_vel
    #     base_lin_vel = robot.data.root_lin_vel_w
    #     base_ang_vel = robot.data.root_ang_vel_w
    #     z_pos = robot.data.root_pos_w[:, 2:3]
        
    #     # Foot positions in world frame
    #     foot_pos = robot.data.body_pos_w[:, self._amp_foot_body_indices, :] - robot.data.root_pos_w[:, :].unsqueeze(1)  #robot.data.body_pos_w[:, self._amp_foot_body_indices, :]
    #     foot_pos = foot_pos.flatten(1)  # (num_envs, num_feet * 3) 
        
    #     return torch.cat([
    #         joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos
    #     ], dim=-1)

    def get_amp_observations(self) -> torch.Tensor:
        """Get AMP observations for discriminator training.
        
        Returns tensor of shape (num_envs, num_amp_obs):
            joint_pos + base_lin_vel + base_ang_vel + joint_vel
        """
        robot = self.scene.articulations["robot"]
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b

        # Foot positions in local frame
        footpos_translated = robot.data.body_pos_w[:, self._amp_foot_body_indices, :] - robot.data.root_pos_w[:, :].unsqueeze(1)
        footpos_in_body_frame = footpos_translated
        for i in range(len(self._amp_foot_body_indices)):
            footpos_in_body_frame[:,i,:] = math_utils.quat_apply_inverse(robot.data.root_quat_w, footpos_translated[:,i,:])
        #robot.data.body_pos_w[:, self._amp_foot_body_indices, :]
        foot_pos = footpos_in_body_frame.flatten(1)  # (num_envs, num_feet * 3)
        
        return torch.cat([
            joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel
        ], dim=-1)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure Isaac Lab environment for HimLoco RSL-RL."""

from __future__ import annotations

import gymnasium as gym
import torch

from himloco_lab.envs import HimlocoManagerBasedRLEnv
from ..env.vec_env import VecEnv

class HimlocoVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for HimLoco RSL-RL.
    
    This wrapper adapts Isaac Lab HimlocoManagerBasedRLEnv to the interface required by 
    HimLoco's custom RSL-RL implementation.
    
    Key features:
    - Converts TensorDict observations to plain tensors
    - Maintains history buffers for observations
    - Tracks terminated environments and their privileged observations
    - Provides 7-value return format required by HimLoco
    
    .. caution::
        This class must be the last wrapper in the wrapper chain.
    
    Args:
        env: The Isaac Lab HimlocoManagerBasedRLEnv environment.
        history_length: Number of historical time steps to stack with current observation.
            0 = only current step, 1 = current + 1 past step, etc.
        privileged_history_length: Number of historical time steps to stack with current privileged observation.
            0 = only current step, 1 = current + 1 past step, etc.
    
    Raises:
        ValueError: When the environment is not an instance of HimlocoManagerBasedRLEnv.
    """

    def __init__(
        self, 
        env: HimlocoManagerBasedRLEnv,
        history_length: int = 0,
        privileged_history_length: int = 0,
    ):
        """Initialize the wrapper.
        
        Args:
            env: Isaac Lab HimlocoManagerBasedRLEnv environment.
            history_length: Number of historical time steps to stack with current observation.
                0 = only current step (num_obs = num_one_step_obs)
                1 = current + 1 past step (num_obs = num_one_step_obs * 2)
                n = current + n past steps (num_obs = num_one_step_obs * (n+1))
            privileged_history_length: Number of historical time steps to stack with current privileged observation.
                Same interpretation as history_length.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, HimlocoManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from HimlocoManagerBasedRLEnv. "
                f"Environment type: {type(env)}"
            )
        
        # check that observation manager exists
        if not hasattr(env.unwrapped, "observation_manager"):
            raise ValueError(
                "The environment must have an observation_manager. "
                "HimlocoManagerBasedRLEnv requires observation_manager to be configured."
            )
        # check that action manager exists
        if not hasattr(env.unwrapped, "action_manager"):
            raise ValueError(
                "The environment must have an action_manager. "
                "HimlocoManagerBasedRLEnv requires action_manager to be configured."
            )
        
        # initialize the wrapper
        self.env = env
        
        # store information required by HimLoco RSL-RL
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        
        # obtain action dimension from action manager
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        
        # obtain single-step observation dimensions from observation manager
        # These are the dimensions of observations returned by the environment at each step
        self.num_one_step_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        
        # Set history lengths
        self.history_length = history_length
        
        # Calculate total observation dimensions
        # history_length=0 means only current step (num_obs = num_one_step_obs)
        # history_length=1 means current + 1 past step (num_obs = num_one_step_obs * 2)
        self.num_obs = self.num_one_step_obs * (self.history_length + 1)
        
        # History buffer for policy observations
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        
        # Buffers for terminated environments (HimLoco-specific)
        self._termination_ids = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Initialize privileged observation related attributes if critic observations exist
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_one_step_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
            self.privileged_history_length = privileged_history_length
            self.num_privileged_obs = self.num_one_step_privileged_obs * (self.privileged_history_length + 1)
            self.privileged_obs_history_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
            self._termination_privileged_obs = torch.zeros(0, self.num_privileged_obs, device=self.device)
        else:
            self.num_one_step_privileged_obs = None
            self.privileged_history_length = None
            self.num_privileged_obs = None
            self.privileged_obs_history_buf = None
            self._termination_privileged_obs = None
        
        # reset at the start since the HimLoco runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.env.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self):
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def dt(self) -> float:
        """Time between environment steps (decimation * physics_dt)."""
        return self.unwrapped.step_dt

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        """Joint position limits from the articulation. Shape: (num_dof, 2)."""
        robot = self.unwrapped.scene.articulations["robot"]
        return robot.data.soft_joint_pos_limits[0]  # [num_dof, 2]

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in HimLoco RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    def get_amp_observations(self) -> torch.Tensor:
        """Get AMP observations from the underlying environment."""
        return self.unwrapped.get_amp_observations()

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        """Set the seed for the environment."""
        return self.env.seed(seed)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reset the environment.
        
        Returns:
            obs policy
        """
        # reset the environment
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def get_observations(self) -> torch.Tensor:
        """Get current policy observations.
        
        Returns:
            Policy observations (history-stacked) as a tensor.
        """
        return self.obs_history_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        """Get current privileged observations.
        
        Returns:
            Privileged observations (history-stacked) as a tensor, or None if not available.
        """
        return self.privileged_obs_history_buf

    def compute_termination_observations(self, env_ids: torch.Tensor, obs_before_reset: torch.Tensor) -> torch.Tensor | None:
        """This method extracts the privileged observations for environments that just terminated.
        
        Args:
            env_ids: Indices of environments that terminated.
            obs_before_reset: observations from environment before reset after action.
        
        Returns:
            Privileged observations for terminated environments.
            Shape: (len(env_ids), num_one_step_privileged_obs).
        """
        if len(env_ids) == 0:
            return torch.zeros(0, self.num_one_step_privileged_obs, device=self.device)
        
        termination_obs = obs_before_reset[env_ids]
        
        return termination_obs

    def step(self, actions: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, dict,
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Execute one time-step of the environment's dynamics.
        
        This extends the standard step to track terminated environments and their
        privileged observations, which is required by HimLoco for proper value
        bootstrapping at episode boundaries.
        
        Args:
            actions: Actions to apply in the environment.
        
        Returns:
            obs: Policy observations for all environments
            privileged_obs: Privileged/critic observations (None if not available)
            rewards: Rewards for all environments
            dones: Done flags for all environments
            infos: Additional information dictionary
            termination_ids: Indices of environments that terminated this step
            termination_privileged_obs: Privileged obs for terminated environments
            terminal_amp_states: AMP observations for terminated environments (pre-reset).
        """
        # execute step in Isaac Lab environment (no action clipping, managed by environment)
        obs_dict, obs_before_reset, rewards, terminated, truncated, infos, \
            _, _, terminal_amp_states = self.env.step(actions)
        
        # compute dones for compatibility with HimLoco RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        
        # move time out information to the extras dict (for infinite horizon tasks)
        if not self.unwrapped.cfg.is_finite_horizon:
            infos["time_outs"] = truncated
        
        # Extract policy observations (single-step from environment)
        if "policy" in obs_dict:
            current_obs = obs_dict["policy"]
        else:
            first_key = next(iter(obs_dict.keys()))
            current_obs = obs_dict[first_key]
        
        # Update policy observation history buffer
        # Environment returns single-step obs, wrapper does history stacking
        # [new_single_step_obs, old_history[:-single_step]]
        if self.history_length > 0:
            self.obs_history_buf = torch.cat(
                (
                    current_obs[:, :self.num_one_step_obs],
                    self.obs_history_buf[:, :-self.num_one_step_obs]
                ),
                dim=-1
            )
        else:
            self.obs_history_buf = current_obs
        
        # Track which environments terminated this step (HimLoco-specific)
        self._termination_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        
        # Update privileged observation history buffer if available
        if "critic" in obs_dict:
            current_privileged_obs = obs_dict["critic"]
            # Compute termination observations before updating buffer
            termination_observation = obs_before_reset["critic"]
            self._termination_privileged_obs = self.compute_termination_observations(
                self._termination_ids, termination_observation
            )
            # Update history buffer
            if self.privileged_history_length > 0:
                self.privileged_obs_history_buf = torch.cat(
                    (
                        current_privileged_obs[:, :self.num_one_step_privileged_obs],
                        self.privileged_obs_history_buf[:, :-self.num_one_step_privileged_obs]
                    ),
                    dim=-1
                )
            else:
                self.privileged_obs_history_buf = current_privileged_obs
        
        # Check for NaN/Inf in observations before returning
        if torch.isnan(self.obs_history_buf).any() or torch.isinf(self.obs_history_buf).any():
            raise ValueError("NaN/Inf detected in obs_history_buf!")
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            raise ValueError("NaN/Inf detected in rewards!")
            
        return (
            self.obs_history_buf,
            self.privileged_obs_history_buf,
            rewards,
            dones,
            infos,
            self._termination_ids,
            self._termination_privileged_obs,
            terminal_amp_states,
        )

    def close(self):
        """Close the environment."""
        return self.env.close()

    def __getattr__(self, name):
        """Forward all other attribute access to wrapped environment."""
        return getattr(self.env, name)

    
    def get_clean_observations(self) -> torch.Tensor:
        """
        提取当前步无噪声单帧本体观测 \bar{o}_t (45维)，严格对齐 ecow PolicyCfg 的缩放系数 (0.25 与 0.05):
        [velocity_commands(3), base_ang_vel*0.25(3), projected_gravity(3),
         joint_pos_rel(12), joint_vel_rel*0.05(12), last_action(12)]
        """
        robot = self.unwrapped.scene["robot"]
        cmd = self.unwrapped.command_manager.get_command("base_velocity")[:, :3]
        base_ang_vel = robot.data.root_ang_vel_b * 0.25
        proj_gravity = robot.data.projected_gravity_b
        dof_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        dof_vel_rel = (robot.data.joint_vel - robot.data.default_joint_vel) * 0.05
        last_action = self.unwrapped.action_manager.action

        return torch.cat(
            [cmd, base_ang_vel, proj_gravity, dof_pos_rel, dof_vel_rel, last_action], dim=-1
        ).to(device=self.device, dtype=torch.float32)

    def update_clean_obs_history(self, reset_env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """维护无噪声 3D 历史观测队列 [num_envs, 6, 45]，第 0 帧始终为最新帧 o_t"""
        clean_obs = self.get_clean_observations()
        single_obs_dim = clean_obs.shape[-1]
        hist_len = self.num_obs // single_obs_dim

        if not hasattr(self, "_clean_obs_hist_3d") or self._clean_obs_hist_3d.shape[0] != self.num_envs:
            self._clean_obs_hist_3d = torch.zeros(
                self.num_envs, hist_len, single_obs_dim, device=self.device, dtype=torch.float32
            )
            self._clean_obs_hist_3d[:, 0, :] = clean_obs
            return self._clean_obs_hist_3d.clone()

        if reset_env_ids is not None and reset_env_ids.numel() > 0:
            self._clean_obs_hist_3d[reset_env_ids] = 0.0

        self._clean_obs_hist_3d = torch.cat(
            [clean_obs.unsqueeze(1), self._clean_obs_hist_3d[:, :-1, :]], dim=1
        )
        return self._clean_obs_hist_3d.clone()

    def get_obs_history_3d(self) -> torch.Tensor:
        if not hasattr(self, "_clean_obs_hist_3d"):
            return self.update_clean_obs_history()
        return self._clean_obs_hist_3d.clone()

    def get_wm_state(self) -> torch.Tensor:
        """
        提取 46 维无冗余统一世界模型状态 s_t:
        [root_z(1), base_lin_vel(3), base_ang_vel(3), proj_gravity(3),
         dof_pos_rel(12), dof_vel_rel(12), foot_pos_b(12)]
        """
        robot = self.unwrapped.scene["robot"]
        root_z = robot.data.root_pos_w[:, 2:3]
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b
        proj_gravity = robot.data.projected_gravity_b
        dof_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        dof_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel
        amp_obs = self.get_amp_observations()
        foot_pos_b = amp_obs[:, 12:24]

        return torch.cat(
            [root_z, base_lin_vel, base_ang_vel, proj_gravity, dof_pos_rel, dof_vel_rel, foot_pos_b], dim=-1
        ).to(device=self.device, dtype=torch.float32)

    def wm_state_to_amp_obs(self, wm_state: torch.Tensor) -> torch.Tensor:
        """
        从 46 维世界模型状态直接可微组装 42 维 AMP 观测:
        [joint_pos(12), foot_pos_b(12), base_lin_vel(3), base_ang_vel(3), joint_vel(12)]
        确保 AMP 判别器的梯度 100% 传导至与 Actor / Task Reward 完全相同的状态变量！
        """
        robot = self.unwrapped.scene["robot"]
        default_pos = robot.data.default_joint_pos[:1, :].to(wm_state.device)
        default_vel = robot.data.default_joint_vel[:1, :].to(wm_state.device)

        joint_pos = wm_state[:, 10:22] + default_pos
        foot_pos_b = wm_state[:, 34:46]
        base_lin_vel = wm_state[:, 1:4]
        base_ang_vel = wm_state[:, 4:7]
        joint_vel = wm_state[:, 22:34] + default_vel

        return torch.cat([joint_pos, foot_pos_b, base_lin_vel, base_ang_vel, joint_vel], dim=-1)

    def apply_terminal_amp_to_wm_state(
        self, wm_state: torch.Tensor, termination_ids: torch.Tensor, terminal_amp_states: torch.Tensor
    ) -> torch.Tensor:
        """将 Reset 前的真实终止状态写回 46 维 wm_state，防止跨局重置瞬移污染"""
        if termination_ids.numel() == 0:
            return wm_state
        robot = self.unwrapped.scene["robot"]
        default_pos = robot.data.default_joint_pos[termination_ids].to(wm_state.device)
        default_vel = robot.data.default_joint_vel[termination_ids].to(wm_state.device)

        fixed = wm_state.clone()
        fixed[termination_ids, 10:22] = terminal_amp_states[:, 0:12] - default_pos
        fixed[termination_ids, 34:46] = terminal_amp_states[:, 12:24]
        fixed[termination_ids, 1:4] = terminal_amp_states[:, 24:27]
        fixed[termination_ids, 4:7] = terminal_amp_states[:, 27:30]
        fixed[termination_ids, 22:34] = terminal_amp_states[:, 30:42] - default_vel
        return fixed

    def get_sim_context(self) -> torch.Tensor:
        """提取 6 维域随机化上下文 : [coms_xyz(3), base_mass(1), kp(1), kd(1)]"""
        robot = self.unwrapped.scene["robot"]
        num_envs = self.num_envs

        if not hasattr(self, "_cached_static_context") or self._cached_static_context is None:
            try:
                coms = robot.root_physx_view.get_coms()[:, 0, :3].to(device=self.device, dtype=torch.float32)
            except Exception:
                coms = torch.zeros(num_envs, 3, device=self.device, dtype=torch.float32)
            try:
                masses = robot.root_physx_view.get_masses()[:, 0:1].to(device=self.device, dtype=torch.float32)
            except Exception:
                masses = torch.ones(num_envs, 1, device=self.device, dtype=torch.float32)
            self._cached_static_context = torch.cat([coms, masses], dim=-1)

        try:
            stiffness = robot.data.joint_stiffness[:, :1].to(device=self.device, dtype=torch.float32)
            damping = robot.data.joint_damping[:, :1].to(device=self.device, dtype=torch.float32)
        except Exception:
            stiffness = torch.ones(num_envs, 1, device=self.device, dtype=torch.float32)
            damping = torch.ones(num_envs, 1, device=self.device, dtype=torch.float32)

        return torch.cat([self._cached_static_context, stiffness, damping], dim=-1)

    def get_physics_state(self) -> torch.Tensor:
        """获取当前环境的 37 维物理状态 (SuperMimic WorldModel 格式)"""
        if hasattr(self.unwrapped, "read_physics_state"):
            return self.unwrapped.read_physics_state()
        robot = self.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w - self.unwrapped.scene.env_origins
        return torch.cat(
            (
                root_pos,
                robot.data.root_quat_w,
                robot.data.root_lin_vel_w,
                robot.data.root_ang_vel_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
            ),
            dim=-1,
        ).clone()

    def get_post_step_physics_state(self) -> torch.Tensor:
        """获取 step 后的真实下一物理状态（自动还原被 _reset_idx 覆盖的终止环境状态）"""
        next_state = self.get_physics_state()
        snapshot = getattr(self.unwrapped, "terminal_physics_snapshot", None)
        if snapshot is not None:
            reset_ids, term_state = snapshot
            next_state[reset_ids] = term_state
        return next_state

    def get_velocity_commands(self) -> torch.Tensor:
        """获取当前 [vx, vy, wz] 速度指令 (shape: [num_envs, 3])"""
        return self.unwrapped.command_manager.get_command("base_velocity").clone()

    def get_feet_contact(self) -> torch.Tensor:
        """获取四足触地布尔状态 (shape: [num_envs, 4])"""
        contact_sensor = self.unwrapped.scene.sensors["contact_forces"]
        if not hasattr(self, "_foot_body_ids"):
            self._foot_body_ids, _ = contact_sensor.find_bodies(".*_foot")
        forces_z = contact_sensor.data.net_forces_w[:, self._foot_body_ids, 2]
        return (torch.abs(forces_z) > 1.0).float().clone()

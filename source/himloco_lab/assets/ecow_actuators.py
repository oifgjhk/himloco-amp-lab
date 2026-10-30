from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.buffers import DelayBuffer


class EcowActuator(IdealPDActuator):
    """Panda actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    X2: No-Load Speed Test
    """

    cfg: PandaActuatorCfg

    armature: torch.Tensor
    """The armature of the actuator joints. Shape is (num_envs, num_joints).
        armature = J2 + J1 * i2 ^ 2 + Jr * (i1 * i2) ^ 2
    """

    def __init__(self, cfg: PandaActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        assert cfg.X1 < cfg.X2, "X1 must be less than X2"
        assert cfg.Y1 <= cfg.Y2, "Y1 must be less than or equal to Y2"

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = torch.tensor(cfg.Y1, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._effort_y2 = torch.tensor(cfg.Y2, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._velocity_x1 = torch.tensor(cfg.X1, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._velocity_x2 = torch.tensor(cfg.X2, dtype=self.computed_effort.dtype, device=self.computed_effort.device)

        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
    
    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # calculate the desired joint torques
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class EcowActuatorCfg(IdealPDActuatorCfg):
    """
    Configuration for Ecow actuators.
    """

    class_type: type = EcowActuator

    X1: float = MISSING
    """Maximum Speed at Full Torque(T-N Curve Knee Point) Unit: rad/s"""

    X2: float = MISSING
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = MISSING
    """Peak Torque Test(Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float = MISSING
    """Peak Torque Test(Torque and Speed in the Opposite Direction) Unit: N*m"""

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@configclass
class EcowActuatorCfg_HIP(EcowActuatorCfg):
    X1 = 6.96386
    X2 = 12.8805
    Y1 = 1500 # 854
    Y2 = 1500 # 854
    armature = 1.103652102
    min_delay = 0
    max_delay = 5


@configclass
class EcowActuatorCfg_THIGH(EcowActuatorCfg):
    X1 = 7.4351
    X2 = 13.823
    Y1 = 1400 # 824
    Y2 = 1400 # 824
    armature = 1.076606799
    min_delay = 0
    max_delay = 5

@configclass
class EcowActuatorCfg_CALF(EcowActuatorCfg):
    X1 = 5.235988
    X2 = 9.686577
    Y1 = 2000 # 1148
    Y2 = 2000 # 1148
    armature = 2.00462812
    min_delay = 0
    max_delay = 5
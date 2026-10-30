from dataclasses import MISSING

from isaaclab.utils import configclass

from .commands import UniformLevelVelocityCommand

from isaaclab.envs.mdp import UniformVelocityCommandCfg

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    
    class_type: type = UniformLevelVelocityCommand
    
    curriculums_limit_ranges: tuple[float, float] = MISSING

    terrain_limit_ranges: dict[str, tuple[float, float]] = MISSING
    
    low_vel_env_lin_x_ranges: tuple[float, float] = MISSING
    
    rel_high_vel_envs: float = MISSING

    rel_heading_envs: float = MISSING
    
    min_command_norm: float = MISSING

    curriculums_limit_ranges_ang_vel: tuple[float, float] = MISSING

    rel_standing_envs: float = MISSING

    straight_target_prob: float = MISSING

    rotation_target_prob: float = MISSING

    lateral_target_prob: float = MISSING
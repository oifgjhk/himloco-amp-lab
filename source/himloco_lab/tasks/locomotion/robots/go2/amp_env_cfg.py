"""Go2 AMP (Adversarial Motion Priors) environment configuration.

AMP modifies the training algorithm (runner + discriminator), not the environment itself,
so this simply reuses the velocity-tracking environment configuration.
"""

from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg  # noqa: F401

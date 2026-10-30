import gymnasium as gym

gym.register(
    id="Ecow-Velocity",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"
    },
)

gym.register(
    id="Ecow-Velocity-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Ecow-AMP",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:RobotEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_amp_rsl_rl_cfg_ecow:AMPPPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

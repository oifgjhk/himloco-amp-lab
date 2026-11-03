"""HIMLOCO AMP PPO 配置"""

from isaaclab.utils import configclass
from himloco_lab.rsl_rl.config import (
    AMPOnPolicyRunnerCfg,
    AMPPPOAlgorithmCfg,
    AMPPPOActorCriticCfg,
)

# Resolve motion files at import time (absolute paths)
# agents → locomotion → tasks → himloco_lab → source → project_root
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_MOTION_DIR = _PROJECT_ROOT / "datasets" / "motions" / "ecow_new"
_AMP_MOTION_FILES = sorted(str(p) for p in _MOTION_DIR.glob("*.json"))


@configclass
class AMPPPORunnerCfg(AMPOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500000
    save_interval = 200
    experiment_name = "ecow_amp"
    history_length = 5
    policy = AMPPPOActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        activation='elu',
        fixed_std=False,
    )
    algorithm = AMPPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule='adaptive',
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size=1000000,
    )
    amp_motion_files = _AMP_MOTION_FILES
    amp_reward_coef = 0.05 #0.06 #0.03 # #0.3 #0.5 * 0.02 #0.3
    """Coefficient for AMP style reward."""
    amp_task_reward_lerp = 0.7 #0.5 #0.3 #0.7
    """Ratio to lerp between task reward and style reward."""
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    # --- 新增训练模式切换开关 ---
    train_mode: str = "bptt"             # 可选: "ppo" (传统无模型PPO) | "bptt" (SuperMimic 世界模型+BPTT)
    
    # --- SuperMimic 世界模型与 BPTT 超参数 (对照论文 Table A.5) ---
    wm_window_len: int = 8               # k = 8 步自回归拟合窗口
    bptt_horizon: int = 32               # H = 32 步想象滚动时域
    wm_lr: float = 2e-4                  # 世界模型与 BPTT 学习率 2e-4
    wm_buffer_capacity: int = 1000       # 每个环境存储 1000 步历史
    wm_batches_per_update: int = 10      # 每次更新采样 10 个 batch (每个 2048 窗口)
    wm_batch_size: int = 2048
    bptt_batch_size: int = 4096          # 每次 BPTT 采样 4096 条想象轨迹
    bptt_action_noise_std: float = 0.1   # 仿真采样时的探索噪声 N(0, 0.1^2)
    bptt_sim_steps_per_update: int = 2   # 对应论文 UTD (Update-to-Data ratio) = 0.4 ~ 0.5

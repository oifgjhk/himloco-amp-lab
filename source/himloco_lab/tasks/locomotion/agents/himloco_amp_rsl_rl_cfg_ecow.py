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

    # ==================== 训练模式切换开关 ====================
    # 可选: "bptt" (SuperMimic 世界模型 + BPTT 想象训练) | "ppo" (传统无模型 AMP-PPO)
    train_mode: str = "bptt"

    # ==================== SuperMimic 世界模型与 BPTT 超参数 ====================
    wm_window_len: int = 8               # k = 8 步自回归世界模型拟合窗口
    bptt_horizon: int = 32               # H = 32 步可微想象滚动时域
    wm_width: int = 2048                 # 世界模型 MLP 隐藏层宽度 (2048x3)
    wm_lr: float = 2.0e-4                # 世界模型学习率
    actor_lr: float = 2.0e-4             # BPTT Actor 初始学习率
    critic_lr: float = 2.0e-4            # BPTT Critic 初始学习率
    discriminator_lr: float = 1.0e-5     # BPTT 模式下 AMP 判别器学习率 (对齐 SuperMimic)
    lr_decay_updates: int = 250000       # 余弦学习率衰减总步数
    wm_buffer_capacity: int = 256        # 每个环境在 ReplayBuffer 中存储的时间步容量
    wm_warmup_steps: int = 64            # 启动预训练前先收集的仿真步数
    wm_pretrain_updates: int = 500       # 仅更新世界模型与判别器的预热迭代次数 (对齐 SuperMimic)
    wm_batches_per_update: int = 10      # 每次迭代更新世界模型的批次数
    wm_batch_size: int = 2048            # 世界模型训练批次大小
    bptt_batch_size: int = 1024          # 每次 BPTT 采样的想象轨迹数量
    bptt_action_noise_std: float = 0.1   # 真实环境采样时的探索噪声标准差
    bptt_sim_steps_per_update: int = 3   # 每次迭代在真实环境中采样的步数 (对应 UTD ≈ 0.33~0.4)
    nominal_height: float = 0.72         # Ecow 标称站立机身高度 (m)
    min_height: float = 0.32             # Ecow 摔倒高度下界 (m)
    max_tilt_deg: float = 65.0           # Ecow 摔倒最大倾角 (deg)

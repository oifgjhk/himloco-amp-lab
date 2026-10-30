"""Critic 特权观测函数 —— 仅 Critic 可见，Actor 拿不到这些信息。

包含两个自定义观测项：
1. base_external_force  — 读取施加在 base 上的合外力（3 维）
2. height_scan_clip     — 地形高度扫描，裁剪到 [-1, 1]
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import RayCaster


def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """观测施加在 base 上的合外力。

    用于 Critic 特权观测，训练时 Critic 知道机器人是否正在被外力推动，
    从而给出更准确的价值估计。Actor 看不到该信息，保证 sim-to-real 可迁移。

    Args:
        env: RL 环境
        asset_cfg: 资产配置，指定观测哪个 body

    Returns:
        (num_envs, 3) 的外力张量 [Fx, Fy, Fz]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: (num_envs, 3)
    return asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :].squeeze(1).clone()


def height_scan_clip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    clip: tuple[float, float] = (-1.0, 1.0),
    offset: float = 0.5) -> torch.Tensor:
    """地形高度扫描，从顶部 ray caster 向下扫描，计算每个扫描点离地高度。

    高度 = 传感器世界坐标 Z - 射线命中点 Z - offset
    结果裁剪到 [clip_min, clip_max] 范围内。
    Critic 通过该信息感知脚下地形起伏，判断当前状态安全性。

    Args:
        env: RL 环境
        sensor_cfg: RayCaster 传感器配置
        clip: 高度裁剪范围，默认 [-1, 1]
        offset: 高度偏移量，默认 0.5

    Returns:
        裁剪后的高度扫描值
    """
    # 提取传感器实例（用于类型提示）
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 高度 = 传感器高度 - 射线命中点高度 - 偏移量
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clip(height, clip[0], clip[1])
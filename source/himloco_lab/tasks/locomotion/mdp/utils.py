from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:
    """计算指定子地形名称在 TerrainGenerator 中的列索引范围 (col_start, col_end)。"""
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device)
    proportions = proportions / proportions.sum()
    cumsum_props = torch.cumsum(proportions, dim=0)

    terrain_idx = sub_terrain_names.index(terrain_name)
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    return (col_start, col_end)


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    """根据 terrain.terrain_types 判断每个环境当前所处的子地形类型。"""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range
    del asset_name
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


def sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """在分离区间 [cfg_min, -limit_bound] U [limit_bound, cfg_max] 上均匀采样。"""
    width_neg = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_pos = torch.nn.functional.relu(cfg_max - limit_bound)

    total_width = width_neg + width_pos + 1e-6
    u = torch.rand(len(env_ids), device=device) * total_width

    samples = torch.where(
        u < width_neg,
        cfg_min + u,
        cfg_max - width_pos + (u - width_neg),
    )
    return samples


def sample_single_interval(env_ids, cfg_min, cfg_max, device):
    """在连续区间 [cfg_min, cfg_max] 上均匀采样。"""
    r = torch.rand(len(env_ids), device=device)
    return cfg_min + r * (cfg_max - cfg_min)
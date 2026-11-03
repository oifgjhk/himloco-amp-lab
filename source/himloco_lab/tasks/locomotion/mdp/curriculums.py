from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np
from .utils import is_robot_on_terrain
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
def get_terrain_mapping(env: ManagerBasedRLEnv) -> tuple[list[str], list[str]]:
    """解析地形配置，获取 sub_terrain_names 列表以及列索引 (terrain_type) 到名称的映射列表"""
    if not (hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "cfg")):
        return [], []

    terrain_cfg = env.scene.terrain.cfg
    if not (hasattr(terrain_cfg, "terrain_generator") and hasattr(terrain_cfg.terrain_generator, "sub_terrains")):
        return [], []

    terrain_gen_cfg = terrain_cfg.terrain_generator
    sub_terrains_cfg = terrain_gen_cfg.sub_terrains
    sub_terrain_names = list(sub_terrains_cfg.keys())
    num_terrain_types = int(terrain_gen_cfg.num_cols)

    # 按照比例计算每一列 (terrain_type) 对应的 sub_terrain (与 commands.py 分配规则完全一致)
    proportions = np.array(
        [sub_terrains_cfg[name].proportion for name in sub_terrain_names],
        dtype=np.float64,
    )
    proportions = proportions / proportions.sum()
    target_cols = proportions * num_terrain_types

    num_cols_per_terrain = np.floor(target_cols).astype(int)
    remainder = target_cols - num_cols_per_terrain
    remaining = num_terrain_types - num_cols_per_terrain.sum()
    if remaining > 0:
        top_idx = np.argsort(-remainder)[:remaining]
        num_cols_per_terrain[top_idx] += 1

    terrain_type_to_name: list[str] = []
    for name, n in zip(sub_terrain_names, num_cols_per_terrain):
        terrain_type_to_name.extend([name] * int(n))

    while len(terrain_type_to_name) < num_terrain_types:
        terrain_type_to_name.append(sub_terrain_names[-1])

    return sub_terrain_names, terrain_type_to_name


def get_terrain_name(env: ManagerBasedRLEnv, command_term, t_idx: int) -> str:
    """获取指定列索引对应的地形名称"""
    _, terrain_type_to_name = get_terrain_mapping(env)
    if 0 <= t_idx < len(terrain_type_to_name):
        return terrain_type_to_name[t_idx]
    return f"Terrain_{t_idx:02d}"

def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    # 如果环境没有包含地形模块，直接返回配置里的基础上限
    if getattr(command_term, "has_terrain", False) is False:
        return torch.tensor(command_term.cfg.ranges.lin_vel_x[1], device=env.device, dtype=torch.float32)

    if not isinstance(env_ids, torch.Tensor):
        env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids_tensor = env_ids.to(device=env.device, dtype=torch.long)

    if len(env_ids_tensor) == 0:
        # 如果当前步没有环境 reset，Fallback 返回所有环境的平均最大速度上限
        all_terrain_types = env.scene.terrain.terrain_types.long()
        return command_term.terrain_vel_current[all_terrain_types, 1].mean()
    
    reset_terrain_types = env.scene.terrain.terrain_types[env_ids_tensor].long()
    unique_terrains = torch.unique(reset_terrain_types)
    if env.common_step_counter % env.max_episode_length == 0:
        sub_terrain_names, terrain_type_to_name = get_terrain_mapping(env)
        
        category_to_envs = {}
        category_to_cols = {}

        for t_type in unique_terrains:
            t_type_int = int(t_type.item())

            if terrain_type_to_name and 0 <= t_type_int < len(terrain_type_to_name):
                t_name = terrain_type_to_name[t_type_int]
                cols = [i for i, name in enumerate(terrain_type_to_name) if name == t_name]
            else:
                t_name = f"Terrain_{t_type_int:02d}"
                cols = [t_type_int]

            if t_name not in category_to_envs:
                category_to_envs[t_name] = []
                category_to_cols[t_name] = cols
            
            t_mask = (reset_terrain_types == t_type_int)
            category_to_envs[t_name].append(env_ids_tensor[t_mask])

        for t_name, env_list in category_to_envs.items():
            t_env_ids = torch.cat(env_list)

            if len(t_env_ids) == 0:
                continue

            t_high_vel_mask = command_term.is_high_vel_env[t_env_ids]
            t_low_vel_mask = ~t_high_vel_mask

            t_low_vel_ids = t_env_ids[t_low_vel_mask.nonzero(as_tuple=True)[0]]
            t_high_vel_ids = t_env_ids[t_high_vel_mask.nonzero(as_tuple=True)[0]]

            reward_low = (
                torch.mean(env.reward_manager._episode_sums[reward_term_name][t_low_vel_ids])
                / env.max_episode_length_s
                if len(t_low_vel_ids) > 0
                else 0.0
            )
            reward_high = (
                torch.mean(env.reward_manager._episode_sums[reward_term_name][t_high_vel_ids])
                / env.max_episode_length_s
                if len(t_high_vel_ids) > 0
                else 0.0
            )

            threshold = reward_term.weight * 0.8

            should_upgrade = (
                (
                    len(t_low_vel_ids) > 0
                    and len(t_high_vel_ids) > 0
                    and reward_low > threshold
                    and reward_high > threshold
                )
                or (len(t_low_vel_ids) > 0 and len(t_high_vel_ids) == 0 and reward_low > threshold)
                or (len(t_low_vel_ids) == 0 and len(t_high_vel_ids) > 0 and reward_high > threshold)
            )

            if should_upgrade:
                delta_command = torch.tensor([-0.2, 0.2], device=env.device)
                cols_to_update = category_to_cols[t_name]

                ref_col = cols_to_update[0]
                current_range = command_term.terrain_vel_current[ref_col]
                limit_range = command_term.terrain_vel_limits[ref_col]

                new_range = torch.clamp(
                    current_range + delta_command,
                    limit_range[0].item(),
                    limit_range[1].item(),
                )

                for col in cols_to_update:
                    command_term.terrain_vel_current[col] = new_range

                print(
                    f"[Curriculum Terrain: {t_name}] ✅ UPGRADED! "
                    f"R_low: {reward_low:.4f}, R_high: {reward_high:.4f} (Thresh: {threshold:.4f}) "
                    f"New Range: [{new_range[0].item():.2f}, {new_range[1].item():.2f}] (Limit: [{limit_range[0].item():.2f}, {limit_range[1].item():.2f}])"
                )

    # ==================== 2. 按 sub_terrain 类型不重复地汇总打印当前速度范围 ====================
    steps_per_iter = getattr(env, "num_steps_per_env", 24) * 100
    current_step = env.common_step_counter

    # 初始化静态变量，记录上一次打印的 iteration 编号
    if not hasattr(lin_vel_cmd_levels, "_last_printed_iter"):
        lin_vel_cmd_levels._last_printed_iter = -1

    current_iter = current_step // steps_per_iter

    # 在满足打印步数且当前步未打印过时，按照 sub_terrain_names 打印汇总
    if current_iter != lin_vel_cmd_levels._last_printed_iter and current_step > 0:
        lin_vel_cmd_levels._last_printed_iter = current_iter

        if getattr(command_term, "has_terrain", False) and hasattr(command_term, "terrain_vel_current"):
            sub_terrain_names, terrain_type_to_name = get_terrain_mapping(env)

            print(
                f"\n================ [Iteration {current_iter:04d} | Step {current_step}] 重置环境对应地形速度指令范围汇总 ================"
            )

            if sub_terrain_names and terrain_type_to_name:
                # 遍历 sub_terrain_names，保证每种地形类型仅打印一次
                for t_name in sub_terrain_names:
                    # 获取映射到该 sub_terrain 的所有列索引 (cols)
                    cols = [i for i, name in enumerate(terrain_type_to_name) if name == t_name]
                    if not cols:
                        continue

                    v_min = command_term.terrain_vel_current[cols, 0].mean().item()
                    v_max = command_term.terrain_vel_current[cols, 1].mean().item()
                    v_limit_min = command_term.terrain_vel_limits[cols, 0].mean().item()
                    v_limit_max = command_term.terrain_vel_limits[cols, 1].mean().item()

                    print(
                        f"  [{t_name:<20s}] | 当前 range: [{v_min:+.2f}, {v_max:+.2f}] m/s "
                        f"| 极限 limit: [{v_limit_min:+.2f}, {v_limit_max:+.2f}] m/s"
                    )
            else:
                num_terrains = command_term.terrain_vel_current.shape[0]
                for t_idx in range(num_terrains):
                    v_min = command_term.terrain_vel_current[t_idx, 0].item()
                    v_max = command_term.terrain_vel_current[t_idx, 1].item()
                    v_limit_min = command_term.terrain_vel_limits[t_idx, 0].item()
                    v_limit_max = command_term.terrain_vel_limits[t_idx, 1].item()
                    print(
                        f"  [Terrain_{t_idx:02d:<15s}] | 当前 range: [{v_min:+.2f}, {v_max:+.2f}] m/s "
                        f"| 极限 limit: [{v_limit_min:+.2f}, {v_limit_max:+.2f}] m/s"
                    )
            print("=" * 95 + "\n")
    max_vel_limits = command_term.terrain_vel_current[:, 1]
    return max_vel_limits.mean()
# def lin_vel_cmd_levels( env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str = "track_lin_vel_xy",) -> torch.Tensor:
#     command_term = env.command_manager.get_term("base_velocity") # 获取速度命令项
#     ranges = command_term.cfg.ranges # 获取速度命令项的范围配置
#     limit_ranges = command_term.cfg.curriculums_limit_ranges # 获取速度命令项的课程限制范围配置
    
#     # 根据环境ID将环境分为低速度组和高速度组
#     low_vel_env_ids = (env_ids >= (env.num_envs * command_term.cfg.rel_high_vel_envs))
#     high_vel_env_ids = (env_ids < (env.num_envs * command_term.cfg.rel_high_vel_envs))
#     low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
#     high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

#     # 获取奖励项配置
#     reward_term = env.reward_manager.get_term_cfg(reward_term_name)
#     # 只有在每个环境的一个episode结束时才更新课程
#     if env.common_step_counter % env.max_episode_length == 0:
#         # 计算低速度组和高速度组的平均奖励
#         reward_low = torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids]) / env.max_episode_length_s if len(low_vel_env_ids) > 0 else 0.0
#         reward_high = torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids]) / env.max_episode_length_s if len(high_vel_env_ids) > 0 else 0.0

#         # 调试打印
#         threshold = reward_term.weight * 0.8
#         print(f"[Curriculum] step={env.common_step_counter}, "
#               f"low_vel_envs={len(low_vel_env_ids)}, high_vel_envs={len(high_vel_env_ids)}, "
#               f"reward_low={reward_low:.4f}, reward_high={reward_high:.4f}, "
#               f"threshold={threshold:.4f}, "
#               f"low_ok={reward_low > threshold}, high_ok={reward_high > threshold}, "
#               f"lin_vel_x_range={ranges.lin_vel_x}")

#         # Only update curriculum if both groups are performing well
#         if (len(low_vel_env_ids) > 0 and len(high_vel_env_ids) > 0 and reward_low > reward_term.weight * 0.8 and reward_high > reward_term.weight * 0.8 ) or \
#         (len(low_vel_env_ids) > 0 and len(high_vel_env_ids) == 0 and reward_low > reward_term.weight * 0.8 ) or \
#         (len(low_vel_env_ids) == 0 and len(high_vel_env_ids) > 0 and reward_high > reward_term.weight * 0.8 ):
#             delta_command = torch.tensor([-0.2, 0.2], device=env.device)
#             ranges.lin_vel_x = torch.clamp(
#                 torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
#                 limit_ranges[0],
#                 limit_ranges[1],
#             ).tolist()
#             print(f"[Curriculum] ✅ UPGRADED! New lin_vel_x range: {ranges.lin_vel_x}")

#     return torch.tensor(ranges.lin_vel_x[1], device=env.device)

def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.curriculums_limit_ranges_ang_vel

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s  # type: ignore

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor] | torch.Tensor:
    """通用的地形等级课程更新函数：
    优先使用移动过程中的最大距离 (max_move_distance) 与累积指令矢量 (commands_xy_accumulation) 判断升降级；
    若底层 CommandTerm 未记录上述属性，则自动回退至 reset 瞬间位移判据。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    terrain_cfg = terrain.cfg.terrain_generator
    if terrain_cfg is None:
        return {"mean": torch.zeros((), device=env.device)}

    command_term = env.command_manager.get_term("base_velocity")
    command = env.command_manager.get_command("base_velocity")

    # 1. 计算有效地形半径 (扣除 sub_terrain_border_width)
    border = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
    terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * border)

    # 2. 获取移动距离：优先使用过程最大距离 max_move_distance，否则回退为瞬间距离
    if hasattr(command_term, "max_move_distance"):
        distance = command_term.max_move_distance[env_ids]
    else:
        distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # 升级条件：过程最大距离超过有效地形边长的一半
    move_up = distance > (terrain_length / 2.0)

    # 3. 计算降级阈值：优先使用 commands_xy_accumulation 矢量和，避免中途反向重采样导致误降级
    if hasattr(command_term, "commands_xy_accumulation"):
        cmd_accum = command_term.commands_xy_accumulation[env_ids]
        # 兼容 resampling_time_range=(min, max) 与标量 resampling_time
        if hasattr(command_term.cfg, "resampling_time_range"):
            r_range = command_term.cfg.resampling_time_range
            resampling_time = 0.5 * (r_range[0] + r_range[1])
        else:
            resampling_time = getattr(command_term.cfg, "resampling_time", 10.0)

        zero_prob = getattr(
            command_term, "zero_command_prob", getattr(command_term.cfg, "rel_standing_envs", 0.0)
        )
        target_dist = torch.norm(cmd_accum, dim=1) * (resampling_time * (1.0 - zero_prob))
    else:
        cmd_lin_speed = (
            torch.abs(command[env_ids, 0])
            if command.shape[-1] == 2
            else torch.norm(command[env_ids, :2], dim=1)
        )
        target_dist = cmd_lin_speed * env.max_episode_length_s

    # 降级条件：过程最大移动距离小于期望净位移的一半，且未满足升级条件
    move_down = (distance < target_dist * 0.5) & (~move_up)

    # 更新地形等级与环境原点
    terrain.update_env_origins(env_ids, move_up, move_down)

    # 4. 统计全局平均等级与各子地形平均等级
    levels = terrain.terrain_levels.float()
    extras: dict[str, torch.Tensor] = {"mean": torch.mean(levels)}
    if terrain_cfg.sub_terrains is not None:
        for terrain_name, sub_cfg in terrain_cfg.sub_terrains.items():
            if getattr(sub_cfg, "proportion", 0.0) <= 0.0:
                continue
            mask = is_robot_on_terrain(env, terrain_name)
            extras[terrain_name] = torch.mean(levels[mask]) if mask.any() else levels.new_zeros(())

    return extras

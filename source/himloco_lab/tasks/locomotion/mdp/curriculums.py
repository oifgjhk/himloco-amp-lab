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
    """严格按照 Isaac Lab TerrainGenerator 的 np.cumsum 规则解析列索引到 sub_terrain 名称的映射"""
    if not (hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "cfg")):
        return [], []

    terrain_cfg = env.scene.terrain.cfg
    if not (hasattr(terrain_cfg, "terrain_generator") and hasattr(terrain_cfg.terrain_generator, "sub_terrains")):
        return [], []

    terrain_gen_cfg = terrain_cfg.terrain_generator
    sub_terrains_cfg = terrain_gen_cfg.sub_terrains
    sub_terrain_names = list(sub_terrains_cfg.keys())
    num_terrain_types = int(terrain_gen_cfg.num_cols)

    proportions = np.array(
        [sub_terrains_cfg[name].proportion for name in sub_terrain_names],
        dtype=np.float64,
    )
    proportions = proportions / proportions.sum()
    cumsum_props = np.cumsum(proportions)

    terrain_type_to_name: list[str] = []
    for col_idx in range(num_terrain_types):
        sub_idx = int(np.min(np.where(col_idx / num_terrain_types + 0.001 < cumsum_props)[0]))
        terrain_type_to_name.append(sub_terrain_names[sub_idx])

    return sub_terrain_names, terrain_type_to_name


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    if getattr(command_term, "has_terrain", False) is False:
        return torch.tensor(command_term.cfg.ranges.lin_vel_x[1], device=env.device, dtype=torch.float32)

    if not isinstance(env_ids, torch.Tensor):
        env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids_tensor = env_ids.to(device=env.device, dtype=torch.long)

    # 1. 在每次有环境 Reset 时，记录这些环境本回合的真实单位时间追踪奖励
    if len(env_ids_tensor) > 0 and hasattr(command_term, "episode_track_rew_ema"):
        ep_Sums = env.reward_manager._episode_sums[reward_term_name][env_ids_tensor]
        # 若环境因走出地形边界或正常超时重置，按实际步数时长归一化；若因摔倒提前终止，按整局时长归一化以维持惩罚
        is_timeout = env.termination_manager.time_outs[env_ids_tensor]
        actual_time_s = (env.episode_length_buf[env_ids_tensor].float() * env.step_dt).clamp(min=2.0)
        denom_time_s = torch.where(
            is_timeout, actual_time_s, torch.full_like(actual_time_s, float(env.max_episode_length_s))
        )
        ep_rew_rate = ep_Sums / denom_time_s

        # 过滤掉训练刚开始第 0 步的初始化 Reset
        valid_ep = env.episode_length_buf[env_ids_tensor] > 10
        if valid_ep.any():
            v_ids = env_ids_tensor[valid_ep]
            v_rates = ep_rew_rate[valid_ep]
            first_time = ~command_term.episode_completed_mask[v_ids]
            command_term.episode_track_rew_ema[v_ids] = torch.where(
                first_time,
                v_rates,
                0.7 * command_term.episode_track_rew_ema[v_ids] + 0.3 * v_rates,
            )
            command_term.episode_completed_mask[v_ids] = True

    # 2. 每隔 max_episode_length 步，汇总全场景各子地形的 EMA 追踪奖励表现进行课程升级
    if env.common_step_counter > 0 and env.common_step_counter % env.max_episode_length == 0:
        sub_terrain_names, terrain_type_to_name = get_terrain_mapping(env)
        all_terrain_types = env.scene.terrain.terrain_types.long()

        for t_name in sub_terrain_names:
            cols = [i for i, name in enumerate(terrain_type_to_name) if name == t_name]
            if not cols:
                continue
            cols_tensor = torch.tensor(cols, device=env.device, dtype=torch.long)
            in_terrain = torch.isin(all_terrain_types, cols_tensor) & command_term.episode_completed_mask
            t_env_ids = in_terrain.nonzero(as_tuple=False).view(-1)
            if len(t_env_ids) == 0:
                continue

            t_high_vel_mask = command_term.is_high_vel_env[t_env_ids]
            t_low_vel_ids = t_env_ids[~t_high_vel_mask]
            t_high_vel_ids = t_env_ids[t_high_vel_mask]

            reward_low = (
                torch.mean(command_term.episode_track_rew_ema[t_low_vel_ids]).item()
                if len(t_low_vel_ids) > 0
                else 0.0
            )
            reward_high = (
                torch.mean(command_term.episode_track_rew_ema[t_high_vel_ids]).item()
                if len(t_high_vel_ids) > 0
                else 0.0
            )

            threshold = reward_term.weight * 0.75

            should_upgrade = (
                (len(t_low_vel_ids) > 0 and len(t_high_vel_ids) > 0 and reward_low > threshold and reward_high > threshold)
                or (len(t_low_vel_ids) > 0 and len(t_high_vel_ids) == 0 and reward_low > threshold)
                or (len(t_low_vel_ids) == 0 and len(t_high_vel_ids) > 0 and reward_high > threshold)
            )

            if should_upgrade:
                delta_command = torch.tensor([-0.2, 0.2], device=env.device)
                ref_col = cols[0]
                current_range = command_term.terrain_vel_current[ref_col]
                limit_range = command_term.terrain_vel_limits[ref_col]
                new_range = torch.clamp(
                    current_range + delta_command,
                    limit_range[0].item(),
                    limit_range[1].item(),
                )
                for col in cols:
                    command_term.terrain_vel_current[col] = new_range
                # 升级后重置该地形的完成标记，确保下一等级重新评估
                command_term.episode_completed_mask[t_env_ids] = False

                print(
                    f"[Curriculum Terrain: {t_name}] ✅ UPGRADED! "
                    f"R_low: {reward_low:.4f}, R_high: {reward_high:.4f} (Thresh: {threshold:.4f}) "
                    f"New Range: [{new_range[0].item():.2f}, {new_range[1].item():.2f}] "
                    f"(Limit: [{limit_range[0].item():.2f}, {limit_range[1].item():.2f}])"
                )

    return command_term.terrain_vel_current[:, 1].mean()
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

# def friction_levels_by_min_terrain(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     init_mu_min: float = 0.60,
#     final_mu_min: float = 0.30,
#     mu_max: float = 1.25,
#     start_terrain_level: float = 3.0,
#     full_terrain_level: float = 7.5,
#     max_step_drop: float = 0.02,
#     cooldown_steps: int = 2000,
# ) -> dict[str, torch.Tensor] | torch.Tensor:
#     """
#     零 PxMaterial 创建的防震荡单向棘轮摩擦课程：
#     - 绝不在运行时调用 root_physx_view.set_material_properties()，彻底杜绝 PhysX 64K 材质上限报错；
#     - 利用地面 friction_combine_mode="multiply" 特性，通过 UsdPhysics.MaterialAPI 原地修改地面唯一材质的缩放倍率。
#     """
#     del env_ids
#     terrain = getattr(env.scene, "terrain", None)
#     if terrain is None or not hasattr(terrain, "terrain_levels"):
#         return torch.tensor(init_mu_min, device=env.device)

#     asset: Articulation = env.scene[asset_cfg.name]
#     terrain_cfg = terrain.cfg.terrain_generator
#     levels = terrain.terrain_levels.float()

#     # 1. 第 0 步初始化：只读获取一次启动时分配给各环境的脚底基础摩擦系数（不创建任何新材质）
#     if not hasattr(env, "_base_foot_friction"):
#         # get_material_properties() 为纯只读操作，不会触发 PxPhysics::createMaterial
#         mat_props = asset.root_physx_view.get_material_properties()  # [num_envs, num_shapes, 3]
#         env._base_foot_friction = mat_props[:, 0, 0].clone().to(env.device)
#         env._curriculum_mu_min = float(init_mu_min)
#         env._terrain_level_watermark = float(start_terrain_level)
#         env._last_mu_update_step = -cooldown_steps
#         env._ground_mat_apis = None

#         # 查找 /World/ground 下的唯一 USD PhysicsMaterial Prim
#         try:
#             from pxr import Usd, UsdPhysics
#             stage = env.sim.stage
#             ground_prim = stage.GetPrimAtPath(terrain.cfg.prim_path)
#             if ground_prim.IsValid():
#                 apis = []
#                 for prim in Usd.PrimRange(ground_prim):
#                     if prim.HasAPI(UsdPhysics.MaterialAPI):
#                         apis.append(UsdPhysics.MaterialAPI(prim))
#                 env._ground_mat_apis = apis
#         except Exception as e:
#             print(f"[Friction Curriculum] Warning: Failed to locate ground UsdPhysics.MaterialAPI: {e}")

#         # 初始地面摩擦放大倍率 = init_mu_min / final_mu_min (例如 0.60 / 0.30 = 2.0)
#         init_scale = float(init_mu_min) / max(float(final_mu_min), 1e-3)
#         if env._ground_mat_apis:
#             for api in env._ground_mat_apis:
#                 api.GetStaticFrictionAttr().Set(init_scale)
#                 api.GetDynamicFrictionAttr().Set(init_scale * 0.95)
#         env.env_friction_coeffs = env._base_foot_friction * init_scale

#     # 2. 计算所有非平地子地形的平均等级最小值 min_terrain_level
#     sub_levels = []
#     if terrain_cfg is not None and terrain_cfg.sub_terrains is not None:
#         for terrain_name, sub_cfg in terrain_cfg.sub_terrains.items():
#             if getattr(sub_cfg, "proportion", 0.0) <= 0.0:
#                 continue
#             if terrain_name == "flat" and len(terrain_cfg.sub_terrains) > 1:
#                 continue
#             mask = is_robot_on_terrain(env, terrain_name)
#             if mask.any():
#                 sub_levels.append(torch.mean(levels[mask]))

#     min_terrain_level = torch.amin(torch.stack(sub_levels)) if len(sub_levels) > 0 else torch.mean(levels)
#     cur_min_lvl = float(min_terrain_level.item())

#     # 3. 水位线单向棘轮更新：仅当短板地形等级创新高且满冷却步数时，原地降低地面唯一材质的摩擦倍率
#     if (
#         cur_min_lvl > env._terrain_level_watermark + 0.05
#         and (env.common_step_counter - env._last_mu_update_step) >= cooldown_steps
#     ):
#         env._terrain_level_watermark = cur_min_lvl
#         env._last_mu_update_step = int(env.common_step_counter)

#         progress = np.clip(
#             (env._terrain_level_watermark - start_terrain_level) / max(full_terrain_level - start_terrain_level, 1e-3),
#             0.0,
#             1.0,
#         )
#         ideal_mu_min = init_mu_min - progress * (init_mu_min - final_mu_min)
#         env._curriculum_mu_min = max(float(ideal_mu_min), env._curriculum_mu_min - max_step_drop, final_mu_min)

#         # 计算地面材质缩放倍率 (从 init_mu_min/final_mu_min 平滑降至 1.0)
#         ground_mu_scale = env._curriculum_mu_min / max(float(final_mu_min), 1e-3)
#         if env._ground_mat_apis:
#             for api in env._ground_mat_apis:
#                 api.GetStaticFrictionAttr().Set(float(ground_mu_scale))
#                 api.GetDynamicFrictionAttr().Set(float(ground_mu_scale * 0.95))

#         # 更新各环境当前的实际有效摩擦系数 (由于地面材质为 multiply 模式，有效摩擦 = 脚底摩擦 * 地面摩擦)
#         env.env_friction_coeffs = env._base_foot_friction * ground_mu_scale

#         print(
#             f"[Friction Curriculum] 📉 Terrain Watermark {env._terrain_level_watermark:.2f} -> "
#             f"Effective mu_min lowered to {env._curriculum_mu_min:.3f} (ground scale: {ground_mu_scale:.2f}x)"
#         )

#     return {
#         "min_terrain_level": min_terrain_level,
#         "terrain_watermark": torch.tensor(env._terrain_level_watermark, device=env.device),
#         "friction_lower_bound": torch.tensor(env._curriculum_mu_min, device=env.device),
#     }

def friction_levels_by_min_terrain(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    init_mu_min: float = 0.60,
    final_mu_min: float = 0.30,
    mu_max: float = 1.25,
    start_terrain_level: float = 3.0,
    min_hold_level: float = 2.0,
    max_step_drop: float = 0.02,
    cooldown_steps: int = 2000,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """
    基于【动态 EMA 止跌企稳判据 + 物理自适应底线】的零材质泄漏摩擦课程：
    1. 彻底移除死板的全局最高水位线 (terrain_watermark)，避免降摩擦后因稳态等级下降而卡死；
    2. 止跌企稳触发机制：
       - 每次降低摩擦后，若地形等级正在下滑 (cur_lvl < level_ema - 0.03)，暂停降摩擦等待策略适应；
       - 一旦地形等级在新摩擦下止跌企稳或回升 (cur_lvl >= level_ema - 0.03)，且高于当前摩擦对应的能力底线，
         即便绝对等级低于高摩擦时期的峰值，在满 cooldown_steps 后仍继续稳步下调摩擦系数；
    3. 仅原地修改地面唯一 USD 材质缩放倍率，零 PxMaterial 创建，彻底规避 PhysX 64K 材质上限报错。
    """
    del env_ids
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        return torch.tensor(init_mu_min, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    terrain_cfg = terrain.cfg.terrain_generator
    levels = terrain.terrain_levels.float()

    # 1. 计算所有非平地子地形的平均等级最小值 min_terrain_level
    sub_levels = []
    if terrain_cfg is not None and terrain_cfg.sub_terrains is not None:
        for terrain_name, sub_cfg in terrain_cfg.sub_terrains.items():
            if getattr(sub_cfg, "proportion", 0.0) <= 0.0:
                continue
            if terrain_name == "flat" and len(terrain_cfg.sub_terrains) > 1:
                continue
            mask = is_robot_on_terrain(env, terrain_name)
            if mask.any():
                sub_levels.append(torch.mean(levels[mask]))

    min_terrain_level = torch.amin(torch.stack(sub_levels)) if len(sub_levels) > 0 else torch.mean(levels)
    cur_min_lvl = float(min_terrain_level.item())

    # 2. 第 0 步初始化：只读记录启动时各环境的脚底基础摩擦系数与地面材质接口
    if not hasattr(env, "_base_foot_friction"):
        mat_props = asset.root_physx_view.get_material_properties()
        env._base_foot_friction = mat_props[:, 0, 0].clone().to(env.device)
        env._curriculum_mu_min = float(init_mu_min)
        env._min_lvl_ema = cur_min_lvl
        env._friction_curriculum_unlocked = False
        env._last_mu_update_step = 0
        env._ground_mat_apis = None

        try:
            from pxr import Usd, UsdPhysics
            stage = env.sim.stage
            ground_prim = stage.GetPrimAtPath(terrain.cfg.prim_path)
            if ground_prim.IsValid():
                env._ground_mat_apis = [
                    UsdPhysics.MaterialAPI(p)
                    for p in Usd.PrimRange(ground_prim)
                    if p.HasAPI(UsdPhysics.MaterialAPI)
                ]
        except Exception as e:
            print(f"[Friction Curriculum] Warning: Failed to locate ground UsdPhysics.MaterialAPI: {e}")

        init_scale = float(init_mu_min) / max(float(final_mu_min), 1e-3)
        if env._ground_mat_apis:
            for api in env._ground_mat_apis:
                api.GetStaticFrictionAttr().Set(init_scale)
                api.GetDynamicFrictionAttr().Set(init_scale * 0.95)
        env.env_friction_coeffs = env._base_foot_friction * init_scale

    # 3. 更新短板地形等级的滑动均值 level_ema（时间常数约 500 步，用于判断当前是“正在掉级”还是“已企稳”）
    env._min_lvl_ema = 0.998 * env._min_lvl_ema + 0.002 * cur_min_lvl

    # 首次达到 start_terrain_level (3.0) 时解锁摩擦课程
    if not env._friction_curriculum_unlocked and cur_min_lvl >= start_terrain_level:
        env._friction_curriculum_unlocked = True
        env._min_lvl_ema = cur_min_lvl

    # 计算当前摩擦系数对应的物理能力底线：
    # 当 mu_min = 0.60 时要求 >= 3.0 级；当 mu_min 降至 0.30 时平滑放宽至 min_hold_level (2.0 级)
    mu_progress = np.clip((init_mu_min - env._curriculum_mu_min) / max(init_mu_min - final_mu_min, 1e-3), 0.0, 1.0)
    required_floor_level = start_terrain_level - mu_progress * (start_terrain_level - min_hold_level)

    # 4. 止跌企稳判断与摩擦步进下调：
    # 条件：(a) 课程已解锁 且 摩擦尚未到底；
    #       (b) 距上次下调已满 cooldown_steps (2000步)；
    #       (c) 当前短板等级高于能力底线 (cur_min_lvl >= required_floor_level)；
    #       (d) 地形等级已止跌企稳或正在上升 (cur_min_lvl >= env._min_lvl_ema - 0.03)
    is_stabilized = cur_min_lvl >= (env._min_lvl_ema - 0.03)
    is_competent = cur_min_lvl >= required_floor_level
    cooldown_ready = (env.common_step_counter - env._last_mu_update_step) >= cooldown_steps

    if (
        env._friction_curriculum_unlocked
        and env._curriculum_mu_min > final_mu_min + 1e-4
        and cooldown_ready
        and is_competent
        and is_stabilized
    ):
        env._curriculum_mu_min = max(final_mu_min, env._curriculum_mu_min - max_step_drop)
        env._last_mu_update_step = int(env.common_step_counter)
        # 重置滑动均值基准为当前等级，下一轮只需在新摩擦的平衡点处企稳即可继续推进
        env._min_lvl_ema = cur_min_lvl

        ground_mu_scale = env._curriculum_mu_min / max(float(final_mu_min), 1e-3)
        if env._ground_mat_apis:
            for api in env._ground_mat_apis:
                api.GetStaticFrictionAttr().Set(float(ground_mu_scale))
                api.GetDynamicFrictionAttr().Set(float(ground_mu_scale * 0.95))

        env.env_friction_coeffs = env._base_foot_friction * ground_mu_scale

        print(
            f"[Friction Curriculum] ✅ Stabilized at level {cur_min_lvl:.2f} (floor: {required_floor_level:.2f}) -> "
            f"mu_min stepped down to {env._curriculum_mu_min:.3f} (ground scale: {ground_mu_scale:.2f}x)"
        )

    return {
        "min_terrain_level": min_terrain_level,
        "level_ema": torch.tensor(env._min_lvl_ema, device=env.device),
        "required_floor": torch.tensor(required_floor_level, device=env.device),
        "friction_lower_bound": torch.tensor(env._curriculum_mu_min, device=env.device),
    }
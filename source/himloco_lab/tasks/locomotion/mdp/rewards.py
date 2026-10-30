from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from himloco_lab.envs import HimlocoManagerBasedRLEnv

"""
关节惩罚.
"""
def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚能耗"""
    asset: Articulation = env.scene[asset_cfg.name] 

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids] # 关节速度
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids] # 关节力矩
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """零命令下惩罚关节位置偏移"""
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
机器人姿态/动作惩罚.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚重力方向与期望方向的偏差、使用L2核函数。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device) # 期望的重力方向
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # 计算当前重力方向与期望重力方向的余弦距离
    normalized = 0.5 * cos_dist + 0.5  # 将余弦距离从[-1, 1]映射到[0, 1]，其中1表示完全对齐，0表示完全相反
    return torch.square(normalized)

def adaptive_orientation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    兼顾平地姿态稳定与地形坡度自适应的重力投影惩罚
    """
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    base_ang_vel = env.scene["robot"].data.root_ang_vel_w

    proj_g_x = projected_gravity[:, 0]  # Pitch 对应分量
    proj_g_y = projected_gravity[:, 1]  # Roll 对应分量

    # 1. Roll 绝对惩罚 (严格限制侧翻)
    roll_penalty = torch.square(proj_g_y)

    # 2. Pitch 小权重的惩罚 (提供平地回正回复力)
    pitch_penalty = torch.square(proj_g_x)

    # 3. 加权汇总：给 Pitch 赋予比 Roll 小得多的权重 (如 0.2 vs 1.0)
    total_penalty = roll_penalty + 0.2 * pitch_penalty

    return total_penalty

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """奖励机器人重力方向与期望方向对齐."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


# def joint_position_penalty(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float,
#     stand_default_joint_pos: list[float] | None = None,
# ) -> torch.Tensor:
#     """惩罚关节位置偏移，零命令或低速时加重惩罚."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :3], dim=1) #torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#     reward = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
#     if stand_default_joint_pos is not None:
#         stand_default_joint_pos = torch.tensor(stand_default_joint_pos, device=env.device)
#         reward_stand = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - stand_default_joint_pos), dim=1)
        
#         return torch.where(cmd > velocity_threshold, reward, stand_still_scale * reward_stand)
#         # return torch.where(torch.logical_or(cmd > velocity_threshold, body_vel > velocity_threshold), reward, stand_still_scale * reward_stand)
#         # return torch.where(body_vel > velocity_threshold, reward, stand_still_scale * reward_stand)
#     else:
#         return torch.where(cmd > velocity_threshold, reward, stand_still_scale * reward)
#         # return torch.where(torch.logical_or(cmd > velocity_threshold, body_vel > velocity_threshold), reward, stand_still_scale * reward)
#         # return torch.where(body_vel > velocity_threshold, reward, stand_still_scale * reward)

def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float,
    stand_default_joint_pos: list[float] | None = None,
) -> torch.Tensor:
    """惩罚关节位置偏移，零命令或低速时加重惩罚."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :3], dim=1) #torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward_walk = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
    if stand_default_joint_pos is not None:
        stand_default_joint_pos = torch.tensor(stand_default_joint_pos, device=env.device)
        reward_stand = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - stand_default_joint_pos), dim=1)
        #print("stand default ", stand_default_joint_pos)
    else:
        reward_stand = reward_walk
        #print("no extra stand default ", asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    # 3. 计算平滑系数 alpha：cmd=0 时为 0.0 (全静止)；cmd >= velocity_threshold 时为 1.0 (全行走)
    alpha = torch.clamp(cmd / velocity_threshold, min=0.0, max=1.0)
    
    # 也可以使用 Smoothstep 使两端导数为 0，过渡更平滑：
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    # 4. 平滑混合：从 (stand_still_scale * reward_stand) 平滑过渡到 (1.0 * reward_walk)
    penalty_stand = stand_still_scale * reward_stand
    penalty_walk = reward_walk
    #print("asset_cfg.joint_ids ", asset_cfg.joint_ids, " joint_pos ", asset.data.joint_pos[:, asset_cfg.joint_ids], " alpha ", alpha)
    
    return (1.0 - alpha) * penalty_stand + alpha * penalty_walk

def smoothness(env: HimlocoManagerBasedRLEnv) -> torch.Tensor:
    """惩罚动作变化率，使用二阶差分。"""
    reward_smoothness = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action*2 + env.pre_pre_action), dim=1)
    return reward_smoothness

def action_rate_l2(env: HimlocoManagerBasedRLEnv) -> torch.Tensor:
    reward_action_rate =  torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward_action_rate
"""
足端奖励.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚足端接触时的水平力过大，可能表示绊倒或碰撞"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 5 * forces_z, dim=1).float()
    return reward

def feet_stumble_continuous(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    连续的足端绊倒惩罚，提供平滑梯度引导机器人脱困
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    
    # 计算超出正常摩擦极限的“碰撞力” 
    stumble_force = forces_xy - 4.0 * forces_z
    
    # 1. 只有超出阈值时才惩罚 (min=0.0)
    # 2. 限制最大值 (max=200.0) 防止刚性碰撞导致的力学数值爆炸破坏网络权重
    penalty = torch.clamp(stumble_force, min=0.0, max=200.0)
    
    # 将四条腿的惩罚值求和并返回
    # 在 reward 配置中，建议将此项的权重设置得小一些 (例如 -0.01 或 -0.005)
    return torch.sum(penalty, dim=1)

def feet_height_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str|None = None,
) -> torch.Tensor:
    """惩罚足端高度偏离目标高度，使用L2核函数，并将位置转换到身体坐标系下计算。零命令时不计算奖励。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    #foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)

    foot_velocity_tanh = torch.tanh(2.0 * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    #reward = torch.sum(foot_z_target_error * foot_leteral_vel, dim=1)
    if command_name is not None:
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward


# def foot_clearance_reward(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
# ) -> torch.Tensor:
#     """惩罚足端未能达到指定高度的地面"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
#     reward = foot_z_target_error * foot_velocity_tanh
#     return torch.exp(-torch.sum(reward, dim=1) / std)

def feet_height(
    env: ManagerBasedRLEnv,
    scan_sensor_names: List[str],
    foot_radius: float,
) -> torch.Tensor:
    """Get the height of the feet above the ground."""
    sensors = [env.scene.sensors[name] for name in scan_sensor_names]
    feet_height = torch.stack(
        [
            sensor.data.pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].squeeze()
            for sensor in sensors
        ],
        dim=-1,
    )
    return feet_height - foot_radius


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scan_sensor_names: List[str],
    foot_radius: float,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        feet_height(env, scan_sensor_names, foot_radius) - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std) * (command_norm > 0.1)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚足端过于接近，可能表示步态不稳定或绊倒"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    奖励在没有命令时足端接触地面。
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)

def feet_air_time_dynamic(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold_range: tuple[float, float] = (0.36, 0.22),
    speed_threshold_range: tuple[float, float] = (0.15, 1.4),
    yaw_threshold_range: tuple[float, float] = (0.15, 1.2),
) -> torch.Tensor:
    """根据速度指令动态调整期望空中滞留时间（Air Time）的奖励函数。

    参数:
        env: 环境实例。
        command_name: 速度指令名称（例如 'base_velocity'）。
        sensor_cfg: 触地传感器配置。
        threshold_range: (最大滞留时间, 最小滞留时间)，单位秒。静止时为 0.36s，全速时为 0.22s。
        speed_threshold_range: (低速死区, 最大线速度)，用于归一化线速度。
        yaw_threshold_range: (低角速度死区, 最大角速度)，用于归一化角速度。
    """
    # 1. 提取传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # 2. 提取并计算指令强度 (cmd_level)
    commands = env.command_manager.get_command(command_name)  # [num_envs, >=3]
    cmd_speed = torch.norm(commands[:, :2], dim=1)
    yaw_speed = torch.abs(commands[:, 2])

    v_min, v_max = speed_threshold_range
    w_min, w_max = yaw_threshold_range

    speed_norm = torch.clamp((cmd_speed - v_min) / (v_max - v_min), 0.0, 1.0)
    yaw_norm = torch.clamp((yaw_speed - w_min) / (w_max - w_min), 0.0, 1.0)
    cmd_level = torch.max(speed_norm, yaw_norm)

    # 3. 动态计算当前的目标滞留时间 (Dynamic Threshold)
    max_th, min_th = threshold_range
    dynamic_threshold = max_th - cmd_level * (max_th - min_th)  # shape: [num_envs]

    # 4. 计算奖励 (当实际滞留时间超过动态阈值时给予正奖励)
    # 扩展维度以支持多足广播 [num_envs, 1]
    reward = torch.sum((last_air_time - dynamic_threshold.unsqueeze(1)) * first_contact, dim=1)

    # 5. 死区过滤：当指令极小时（静止状态）不给予抬腿奖励，防止原地踏步
    is_cmd_active = (cmd_speed > v_min) | (yaw_speed > w_min)
    reward *= is_cmd_active.float()

    return reward
# def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """鼓励四足的步态节拍均匀——不让某只脚在空中/地上停留太久--惩罚项"""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
#     return torch.var(torch.clip(last_air_time, max=1.0), dim=1) + torch.var(
#         torch.clip(last_contact_time, max=1.0), dim=1
#     )

def air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.2,
    min_recorded_time: float = 0.05,
    clip_max_time: float = 1.0,
    ) -> torch.Tensor:
    """Penalize large variance in completed foot swing times during commanded motion.

    This is intentionally mild structure, not a hard gait template. It nudges the
    policy away from messy asymmetric swing timing once a meaningful SE(2)
    locomotion command is present.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :3], dim=1) >= command_threshold

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_air_time = torch.clip(last_air_time, min=0.0, max=clip_max_time)

    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_contact_time = torch.clip(last_contact_time, min=0.0, max=clip_max_time)

    has_recorded_time = torch.logical_and((torch.amax(last_air_time, dim=1) >= min_recorded_time),
    (torch.amax(last_contact_time, dim=1) >= min_recorded_time))

    variance = torch.var(last_air_time, dim=1, correction=0) + torch.var(last_contact_time, dim=1, correction=0)
    return variance * torch.logical_and(moving, has_recorded_time).float()
    
def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """奖励机器人保持在指定高度。

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None: # 如果提供了传感器配置，使用传感器数据调整目标高度以适应地形
        sensor: RayCaster = env.scene[sensor_cfg.name]
        
        ray_hits = sensor.data.ray_hits_w[..., 2]
        
        # 过滤无效值（NaN、Inf、过大值），并将它们替换为NaN以便后续计算忽略
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6) # 根据实际情况调整阈值
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.tensor(float('nan'), device=ray_hits.device)) # 将无效值替换为NaN，以便后续计算中自动忽略这些值
        
        # 调整目标高度：当前目标高度 + 有效传感器读数的平均值（如果所有读数无效，则保持原目标高度不变）
        # 对有效读数取平均，忽略无效值（NaN）
        adjusted_heights = torch.nanmean(ray_hits_masked, dim=1)
        
        # For environments where ALL points are invalid, nanmean returns NaN
        # Replace these with current robot height
        all_invalid_mask = torch.isnan(adjusted_heights)
        if all_invalid_mask.any():
            invalid_env_count = all_invalid_mask.sum().item()
            total_invalid_points = (~valid_mask).sum().item()
            print(f"[WARNING] base_height - {invalid_env_count} envs with all invalid points, total invalid: {total_invalid_points}/{ray_hits.numel()} ({total_invalid_points/ray_hits.numel():.2%})")
            adjusted_heights[all_invalid_mask] = asset.data.root_link_pos_w[all_invalid_mask, 2] - target_height
        
        adjusted_target_height = target_height + adjusted_heights
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def base_height_target_exp(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    std: float = 0.05, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """基于高斯响应的机身高度追踪"""
    """奖励机器人保持在指定高度。

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None: # 如果提供了传感器配置，使用传感器数据调整目标高度以适应地形
        sensor: RayCaster = env.scene[sensor_cfg.name]
        
        ray_hits = sensor.data.ray_hits_w[..., 2]
        
        # 过滤无效值（NaN、Inf、过大值），并将它们替换为NaN以便后续计算忽略
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6) # 根据实际情况调整阈值
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.tensor(float('nan'), device=ray_hits.device)) # 将无效值替换为NaN，以便后续计算中自动忽略这些值
        
        # 调整目标高度：当前目标高度 + 有效传感器读数的平均值（如果所有读数无效，则保持原目标高度不变）
        # 对有效读数取平均，忽略无效值（NaN）
        adjusted_heights = torch.nanmean(ray_hits_masked, dim=1)
        
        # For environments where ALL points are invalid, nanmean returns NaN
        # Replace these with current robot height
        all_invalid_mask = torch.isnan(adjusted_heights)
        if all_invalid_mask.any():
            invalid_env_count = all_invalid_mask.sum().item()
            total_invalid_points = (~valid_mask).sum().item()
            print(f"[WARNING] base_height - {invalid_env_count} envs with all invalid points, total invalid: {total_invalid_points}/{ray_hits.numel()} ({total_invalid_points/ray_hits.numel():.2%})")
            # adjusted_heights[all_invalid_mask] = asset.data.root_link_pos_w[all_invalid_mask, 2] - target_height
            adjusted_heights[all_invalid_mask] = 0.0
        
        adjusted_target_height = target_height + adjusted_heights
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    
    # 偏离目标高度时，奖励快速衰减至 0 (1.0 代表完美匹配)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-height_error / (2 * std ** 2))

# def sound_suppression_acc_per_foot(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """Penalize vertical foot velocity and acceleration during ground contact."""
#     asset = env.scene["robot"]
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     body_ids = sensor_cfg.body_ids
#     foot_vel_z = asset.data.body_vel_w[:, body_ids, 2]
#     foot_acc_z = asset.data.body_acc_w[:, body_ids, 2]
#     contact_force_z = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, body_ids, 2]
#     in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
#     # in_contact = torch.abs(contact_force_z) > 0.0
#     penalty = ((foot_vel_z.square() + foot_acc_z.square()) * in_contact.float()).sum(dim=1)
#     return penalty


# def sound_suppression_acc_per_foot(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     command_name: str = "base_velocity",
# ) -> torch.Tensor:
#     """
#     每只脚单独计算：
#     脚接触地面时，z 方向加速度大/速度大 → 惩罚
#     """

#     asset = env.scene["robot"]
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     body_ids = sensor_cfg.body_ids

#     # 取“脚”的 z 方向线加速度/线速度
#     # shape: (Nenv, Nfeet)
#     foot_vel_z = asset.data.body_vel_w[:, body_ids, 2]
#     foot_acc_z = asset.data.body_acc_w[:, body_ids, 2]

#     # 取脚的接触状态
#     contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
#     in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

#     # 每只脚：加速度平方 × 接触状态
#     acc_penalty = (foot_acc_z ** 2) * in_contact.float()
#     vel_penalty = (foot_vel_z ** 2) * in_contact.float()

#     # 防止数值爆炸
#     acc_penalty = torch.clamp(acc_penalty, max=50.0)
#     vel_penalty = torch.clamp(vel_penalty, max=50.0)

#     # 所有脚加起来
#     penalty = acc_penalty.sum(dim=1) + vel_penalty.sum(dim=1)
#     reward = penalty

#     # # 仅当速度命令较小（小于 1.5）时才启用该奖励
#     # cmd = env.command_manager.get_command(command_name)
    
#     # # 使用 xy 分量的速度范数作为速度大小判断
#     # cmd_speed = torch.norm(cmd[:, :2], dim=1)
#     # reward = reward * (cmd_speed < 1.5).float()

#     return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    asset = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids

    # 1. 提取速度与加速度 (Nenv, Nfeet, 3)
    foot_vel = asset.data.body_vel_w[:, body_ids, :]
    foot_acc = asset.data.body_acc_w[:, body_ids, :]

    # 2. 提取接触状态 (Nenv, Nfeet)
    contact_force_z = contact_sensor.data.net_forces_w[:, body_ids, 2]
    in_contact = (torch.abs(contact_force_z) > 20.0).float()

    # 3. Z 方向惩罚（砸地）
    vel_z_penalty = (torch.clamp(foot_vel[:, :, 2], max = 0) ** 2)
    acc_z_penalty = (foot_acc[:, :, 2] ** 2)

    # 4. XY 方向惩罚（打滑与刮擦）
    # 平面速度模长的平方
    vel_xy_penalty = torch.sum(foot_vel[:, :, :2] ** 2, dim=-1) * in_contact
    acc_xy_penalty = torch.sum(foot_acc[:, :, :2] ** 2, dim=-1) * in_contact

    # 5. 限幅以防数值爆炸
    vel_z_penalty = torch.clamp(vel_z_penalty, max=50.0)
    acc_z_penalty = torch.clamp(acc_z_penalty, max=50.0)
    vel_xy_penalty = torch.clamp(vel_xy_penalty, max=50.0)
    acc_xy_penalty = torch.clamp(acc_xy_penalty, max=50.0)

    # 6. 加权汇总 (可针对 xy 速度和 z 加速度配置不同系数)
    penalty = (
        acc_z_penalty.sum(dim=1) + 
        5.0 * vel_z_penalty.sum(dim=1) + 
        0.5 * vel_xy_penalty.sum(dim=1) +
        0.5 * acc_xy_penalty.sum(dim=1)
    )

    try:
        commands = env.command_manager.get_command(command_name)
        cmd_vel_mag = torch.norm(commands[:, :2], dim=-1) 
        cmd_decay = torch.exp(-cmd_vel_mag / 1.0) 
    except Exception:
        cmd_decay = 1.0

    return penalty * cmd_decay

"""
足部步态.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """奖励足端接触的节拍与期望节拍一致"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


class GaitReward(ManagerTermBase):
    """四足步态强化奖励项.

    他的奖励机制会对在:attr:`synced_feet_pair_names`中定义的选定脚对之间的接触时机差异进行惩罚
    为了使策略偏向于期望的步态，即小跑、弹跳或步态。请注意，此奖励仅用于
    四足步态，具有两对同步移动的脚。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env) # 初始化父类
        self.std: float = cfg.params["std"] # 步态奖励的标准差参数，控制奖励对接触时机差异的敏感度
        self.command_name: str = cfg.params["command_name"] # 命令名称，用于获取命令值
        self.max_err: float = cfg.params["max_err"]     # 步态奖励中用于裁剪接触时机差异的最大误差值，防止过大的误差对奖励产生过度惩罚
        self.velocity_threshold: float = cfg.params["velocity_threshold"] # 速度阈值，只有当机器人速度超过该阈值时才会计算步态奖励，防止在静止或低速时对步态进行过度惩罚
        self.command_threshold: float = cfg.params["command_threshold"] # 命令阈值，只有当命令值超过该阈值时才会计算步态奖励，防止在没有有效命令时对步态进行过度惩罚
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name] # 用于获取足端接触信息的传感器，根据配置中的传感器名称进行查找
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name] # 用于获取机器人状态信息的资产，根据配置中的资产名称进行查找
        # 将脚部身体名称与对应的脚部身体ID匹配
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("这个奖励只支持双脚同步的步态，比如小跑。")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # 对于同步脚，两只脚的接触（空气）时间应匹配
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1]) # 奖励第一对同步脚之间的接触时机差异
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1]) # 奖励第二对同步脚之间的接触时机差异
        sync_reward = sync_reward_0 * sync_reward_1
        # 对于异步脚，一只脚的接触时间应与另一只脚的空气时间匹配
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward
    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

"""
Other rewards.
"""

# # 腿部镜动作镜像惩罚
# def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
#         # Cache joint positions for all pairs
#         env.joint_mirror_joints_cache = [
#             [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
#         ]
#     reward = torch.zeros(env.num_envs, device=env.device)
#     # Iterate over all joint pairs
#     for joint_pair in env.joint_mirror_joints_cache:
#         # Calculate the difference for each pair and add to the total reward
#         reward += torch.sum(
#             torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
#             dim=-1,
#         )
#     reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
#     return reward

def joint_mirror(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    mirror_joints: list[list[str]], 
    mirror_signs: list[float] | None = None
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
        # 如果未指定符号，默认设为 1.0
        if mirror_signs is None:
            env.joint_mirror_signs = [1.0] * len(mirror_joints)
        else:
            env.joint_mirror_signs = mirror_signs

    reward = torch.zeros(env.num_envs, device=env.device)
    # 遍历关节对并应用镜像符号系数
    for idx, joint_pair in enumerate(env.joint_mirror_joints_cache):
        sign = env.joint_mirror_signs[idx]
        diff = asset.data.joint_pos[:, joint_pair[0][0]] - sign * asset.data.joint_pos[:, joint_pair[1][0]]
        reward += torch.sum(torch.square(diff), dim=-1)

    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward

# 依据足端触地情况进行的步态奖励
def gait_type(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5
    fl = contacts[:, 0]
    fr = contacts[:, 1]
    rl = contacts[:, 2]
    rr = contacts[:, 3]

    stand = fl & fr & rl & rr                                     # 四足着地站立
    trot = (fl & rr & ~fr & ~rl) | (fr & rl & ~fl & ~rr)          # 对角小跑

    # gallop = (fl & ~rr & ~fr & ~rl) \
    #        | (~fl & rr & ~fr & ~rl) \
    #        | (~fl & ~rr & fr & ~rl) \
    #        | (~fl & ~rr & ~fr & rl) \
    #        | (~fl & ~rr & ~fr & ~rl)

    # walk = (fl & fr & rl & rr) \
    #      | (fl & fr & ~rl & rr) \
    #      | (~fl & fr & rl & rr) \
    #      | (fl & fr & rl & ~rr) \
    #      | (fl & ~fr & rl & rr)
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd, dim=1)
    ang_vel_cmd = torch.abs(cmd[:, 2])  # |ω_z| yaw方向命令
    trot_mult = torch.where(ang_vel_cmd > 0.15, 1.0, 0.4)  # 转向全权重，直行0.1
    return stand.float() * (cmd_norm < 0.18) * 1.5 + trot.float() * (cmd_norm > 0.22) * trot_mult

# def low_speed(
#     env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_velocity"
# ) -> torch.Tensor:
#     """
#     Rewards or penalizes the robot based on its x-axis speed relative to the commanded speed.
#     根据速度与指令的匹配情况给予奖励或惩罚
#     速度过低：-1.0；速度过高：0.0；速度在理想范围内：2.0；方向相反：-2.0（具有最高优先级，即即使速度在理想范围内，如果方向相反也是-2.0）
#     只在指令的绝对值大于0.1时起作用，只考虑x方向的速度（前进/后退）
#     scale为正数
#     Reward scheme (only active when |command_x| > 0.1):
#     - Speed too low (< 50% of command):  -1.0
#     - Speed too high (> 120% of command):  0.0
#     - Speed in desired range:              2.0
#     - Sign mismatch (wrong direction):    -2.0 (highest priority)

#     This encourages the robot to move at the commanded speed in the correct direction.

#     Returns:
#         A reward tensor with discrete values [-2.0, -1.0, 0.0, 2.0].
#         Returns 0.0 when the command magnitude is below the threshold.

#     Note:
#         scale should be positive.
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)

#     # Calculate the absolute value of speed and command for comparison
#     absolute_speed = torch.abs(asset.data.root_lin_vel_b[:, 0])
#     absolute_command = torch.abs(command[:, 0])

#     # Define speed criteria for desired range
#     speed_too_low = absolute_speed < 0.5 * absolute_command
#     speed_too_high = absolute_speed > 1.2 * absolute_command
#     speed_desired = ~(speed_too_low | speed_too_high)

#     # Check if the speed and command directions are mismatched
#     sign_mismatch = torch.sign(
#         asset.data.root_lin_vel_b[:, 0]
#     ) != torch.sign(command[:, 0])

#     # Initialize reward tensor
#     reward = torch.zeros_like(asset.data.root_lin_vel_b[:, 0])

#     # Assign rewards based on conditions
#     reward[speed_too_low] = -1.0
#     reward[speed_too_high] = 0.0
#     reward[speed_desired] = 2.0
#     # Sign mismatch has the highest priority (overwrites all other assignments)
#     reward[sign_mismatch] = -2.0

#     return reward * (command[:, 0].abs() > 0.1)

def low_speed(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    max_margin_low: float = 0.4,    # 理想区间允许的最大偏低绝对误差 (m/s)
    max_margin_high: float = 0.2,   # 理想区间允许的最大偏高绝对误差 (m/s)
    min_vel_error: float = 0.0,     # 额外惩罚的差值下限
    max_vel_error: float = 1.0,     # 额外惩罚的差值上限 (截断值)
    penalty_scale_low: float = 1.0,
    penalty_scale_high: float = 0.5,
) -> torch.Tensor:
    """
    Rewards or penalizes the robot based on its x-axis speed relative to the commanded speed.
    根据速度与指令的匹配情况给予奖惩：
    - 理想区间：容差同时受【比例 (50%/20%)】与【绝对最大偏差上限 (max_margin)】控制。
      指令较小时采用比例容差，指令较大时容差被限制在固定最大差值内，防止理想区间过宽。
    - 速度偏低/偏高：根据超出理想边界的速度差值（并经 clamp 截断）动态施加线性惩罚。
    - 方向相反：-2.0（最高优先级）。
    - 仅在 |command| > 0.1 时生效。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    v_x = asset.data.root_lin_vel_b[:, 0]
    cmd_x = command[:, 0]

    absolute_speed = torch.abs(v_x)
    absolute_command = torch.abs(cmd_x)

    # 1. 动态计算理想区间的容差边界（结合比例与绝对值上限）
    # 偏低容差：取 50% command 与 max_margin_low 的较小值
    # 偏高容差：取 20% command 与 max_margin_high 的较小值
    margin_low = torch.clamp(0.5 * absolute_command, max=max_margin_low)
    margin_high = torch.clamp(0.2 * absolute_command, max=max_margin_high)

    lower_threshold = absolute_command - margin_low
    upper_threshold = absolute_command + margin_high

    # 2. 速度区间条件判定
    speed_too_low = absolute_speed < lower_threshold
    speed_too_high = absolute_speed > upper_threshold
    speed_desired = ~(speed_too_low | speed_too_high)

    # 3. 计算偏离理想边界的差值，并用 [min_vel_error, max_vel_error] 截断
    diff_low = torch.clamp(lower_threshold - absolute_speed, min=min_vel_error, max=max_vel_error)
    diff_high = torch.clamp(absolute_speed - upper_threshold, min=min_vel_error, max=max_vel_error)

    # 4. 方向异号校验
    sign_mismatch = (v_x * cmd_x) < 0

    # 5. 动态计算奖惩
    reward = torch.zeros_like(v_x)

    # 理想区间：固定奖励
    reward[speed_desired] = 2.0

    # 速度偏低：基础惩罚 -1.0 + 按超出阈值的差值线性增加惩罚
    reward[speed_too_low] = -1.0 - penalty_scale_low * diff_low[speed_too_low]

    # 速度偏高：基础惩罚 0.0 + 按超出阈值的差值线性增加惩罚
    reward[speed_too_high] = 0.0 - penalty_scale_high * diff_high[speed_too_high]

    # 方向相反：最高优先级覆盖
    reward[sign_mismatch] = -2.0

    # 6. 仅在指令绝对值 > 0.1 时生效
    active_mask = absolute_command > 0.1
    return reward * active_mask

# def contact_forces_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """鼓励四足的force均匀——不让某只脚在空中/地上停留太久--惩罚项"""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
#     reward = torch.var(net_contact_forces
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
#     return torch.var(torch.clip(last_air_time, max=1.0), dim=1) + torch.var(
#         torch.clip(last_contact_time, max=1.0), dim=1
#     )

# def contact_forces_variance_penalty(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """惩罚四足接触力及空中/接触时间的不均匀性。"""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history[
#         :, :, sensor_cfg.body_ids
    
#     is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

#     contact_force_z = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, body_ids, 2]
#     # [env, history, feet, 3] -> [env, history, feet]
#     contact_force_magnitudes = torch.linalg.vector_norm(
#         net_contact_forces, dim=-1
#     )
#     force_variance = torch.var(
#         contact_force_magnitudes, dim=-1
#     ).mean(dim=1)

#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

#     air_time_variance = torch.var(
#         torch.clamp(last_air_time, max=1.0), dim=1
#     )
#     contact_time_variance = torch.var(
#         torch.clamp(last_contact_time, max=1.0), dim=1
#     )

#     return force_variance*0.00001 + air_time_variance + contact_time_variance

def stand_z_force_variance_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    command_name: str = "base_velocity",  
    stand_threshold: float = 0.1             
) -> torch.Tensor:
    """
    仅在站立指令下，直接惩罚四足 Z 轴受力方差。
    强制机器人四足均匀触地，防止抬起单腿。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取 4 只脚的 Z 轴受力 [env, 4]
    force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    force_z = torch.clamp(force_z, min=0.0) 
    
    # 【核心修改】直接对全部 4 只脚计算方差！
    # 悬空的脚（受力0）会产生巨大的方差，从而被强烈惩罚
    absolute_variance = torch.var(force_z, dim=-1)
    
    # 引入指令判断，仅在站立时生效
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(commands, dim=1) #torch.linalg.vector_norm(commands[:, :3], dim=-1)
    
    # 速度指令 < 0.1 时，掩码为 1 (站立状态)，否则为 0 (行走状态)
    is_stand_mask = (cmd_norm < stand_threshold).float()
    
    # 返回最终惩罚
    return absolute_variance * is_stand_mask

def base_z_acceleration_penalty(
    env: ManagerBasedRLEnv,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚机身(Base)在 Z 轴方向的线性加速度。
    抑制机器人行走时的剧烈上下起伏和弹跳。
    """
    # 从环境中获取机器人资产 (Articulation)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取世界坐标系下的机身根节点加速度 [env, 3]
    # 物理引擎通常包含 x, y, z 三个方向的线性加速度
    # 提取 Z 轴加速度 [env]
    base_acc_z = asset.data.body_acc_w[:, 0, 2]
    
    # 使用平方 (L2 norm) 计算惩罚。
    # 这样可以忽略微小的起伏，但对剧烈的弹跳给予极大的惩罚
    return torch.square(base_acc_z)


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ,command_name: str = "base_velocity"
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # # Penalize feet sliding
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # asset: RigidObject = env.scene[asset_cfg.name]

    # cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
    #     :, :
    # ].unsqueeze(1)
    # footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    # for i in range(len(asset_cfg.body_ids)):
    #     footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
    #         asset.data.root_quat_w, cur_footvel_translated[:, i, :]
    #     )
    # foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
    #     env.num_envs, -1
    # )
    # reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # return reward

    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (command_norm > 0.1)
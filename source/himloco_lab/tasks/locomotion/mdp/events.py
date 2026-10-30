import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


from isaaclab.envs import ManagerBasedEnv


def apply_periodic_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    period_step: int,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """施加周期性外力外扭矩（域随机化）。

    每隔 period_step 次调用才真正施加一次随机力/力矩，其余时刻施力为零。
    调用计数通过环境属性记录，在第一次调用时自动初始化。

    Args:
        env: RL 环境
        env_ids: 要施力的环境 ID。若为 None，则对所有环境施力。
        period_step: 施力周期（仿真步数）。调用次数达到该值的整数倍时才施力。
        force_range: 力幅值范围 (min, max)，单位 N
        torque_range: 力矩幅值范围 (min, max)，单位 N·m
        asset_cfg: 施力目标资产的配置
    """
    # 首次调用时初始化计数器
    if not hasattr(env, "external_force_call_count") or env.external_force_call_count is None:
        env.external_force_call_count = 0

    # 提取资产对象（用于类型提示）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析环境 ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 解析刚体数量
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    # 创建零力/零力矩张量
    size = (len(env_ids), num_bodies, 3)
    zero_forces = torch.zeros(size, device=asset.device, dtype=torch.float32)
    zero_torques = torch.zeros(size, device=asset.device, dtype=torch.float32)

    # 仅在调用次数达到 period_step 整数倍时施力
    if env.external_force_call_count % period_step == 0:
        # 随机采样力/力矩
        forces = math_utils.sample_uniform(*force_range, size, asset.device)
        torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    else:
        # 其余时刻施力为零
        forces = zero_forces
        torques = zero_torques

    # 将力/力矩写入缓冲区
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )

    # 调用计数加一
    env.external_force_call_count += 1
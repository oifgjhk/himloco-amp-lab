"""Replay and export AMP motion trajectories in Isaac Sim."""

import sys
import os
import json
from pathlib import Path
from isaaclab.app import AppLauncher

import argparse

# local imports
import himloco_rsl_rl.cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Replay and export AMP trajectories in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--env_cfg_entry_point", type=str, default="play_env_cfg_entry_point",
    help="Name of the play environment configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# ---- 新增：轨迹导出相关参数 ----
parser.add_argument("--export", action="store_true", default=False, help="Whether to export simulated trajectories to JSON files.")
parser.add_argument("--output_dir", type=str, default="./sim_exported_motions", help="Directory to save exported trajectory files.")
parser.add_argument("--export_source", type=str, choices=["sim", "amp"], default="sim", 
                    help="Source of exported state: 'sim' (actual Isaac physical state) or 'amp' (resampled AMP target state).")

# append HimLoco RSL-RL cli arguments
himloco_rsl_rl.cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_inv
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl.datasets.motion_loader import AMPLoader
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import isaaclab.terrains as terrain_gen

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)

def extract_sim_state_as_amp_frame(robot, foot_ids, root_pos, root_rot, lin_vel, ang_vel):
    """从 Isaac Sim 的机器人状态 (robot.data) 提取并拼装 61 维 AMP 格式数据帧。
    
    返回维度: (61,) numpy array
    数据排布: [root_pos(3), root_rot_xyzw(4), joint_pos_pb(12), foot_pos_b(12),
              lin_vel_b(3), ang_vel_b(3), joint_vel_pb(12), foot_vel_b(12)]
    """
    # 1. Root Position & Rotation
    root_pos = root_pos[0] #robot.data.root_pos_w[0].detach()  # (3,)
    root_quat_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)[0] #robot.data.root_quat_w[0].detach()  # (4,) [w, x, y, z]
    # 转换为 AMP 的 [x, y, z, w] 四元数格式
    root_rot_xyzw = torch.cat([root_quat_wxyz[1:4], root_quat_wxyz[0:1]], dim=-1)
    # 2. Joint Positions & Velocities (Isaac Lab -> PyBullet/AMP 顺序重排)
    # Isaac: [hips(4), thighs(4), calves(4)] -> PyBullet: [FL, FR, RL, RR]
    joint_pos_isaac = robot.data.joint_pos[0].detach()  # (12,)
    joint_vel_isaac = robot.data.joint_vel[0].detach()  # (12,)
    joint_pos_pb = joint_pos_isaac.view(3, 4).T.reshape(-1)
    joint_vel_pb = joint_vel_isaac.view(3, 4).T.reshape(-1)

    # 3. Root Linear & Angular Velocities (局部机体系)
    lin_vel_b = lin_vel[0] #robot.data.root_lin_vel_b[0].detach()  # (3,)
    ang_vel_b = ang_vel[0] #robot.data.root_ang_vel_b[0].detach()  # (3,)

    # 4. Foot Positions & Velocities in Local Body Frame
    num_feet = len(foot_ids)
    sim_foot_pos_w = robot.data.body_pos_w[0, foot_ids, :]  # (4, 3)
    sim_foot_vel_w = robot.data.body_lin_vel_w[0, foot_ids, :]  # (4, 3)

    # 相对位置与相对线速度
    rel_foot_pos_w = sim_foot_pos_w - root_pos.unsqueeze(0)
    rel_foot_vel_w = sim_foot_vel_w - robot.data.root_lin_vel_w[0].unsqueeze(0)

    # 逆旋转回 Body 坐标系 (P_local = R_root^-1 * P_rel)
    quat_inv_repeat = quat_inv(root_quat_wxyz.unsqueeze(0).repeat(num_feet, 1))
    foot_pos_b = quat_apply(quat_inv_repeat, rel_foot_pos_w).view(-1)  # (12,)
    foot_vel_b = quat_apply(quat_inv_repeat, rel_foot_vel_w).view(-1)  # (12,)

    # 拼接为 61 维数组
    frame_tensor = torch.cat([
        root_pos, root_rot_xyzw, joint_pos_pb, foot_pos_b,
        lin_vel_b, ang_vel_b, joint_vel_pb, foot_vel_b
    ], dim=-1)

    return frame_tensor.cpu().numpy()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    agent_cfg = himloco_rsl_rl.cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # flat plane, no terrain generator
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG

    # disable all domain randomization events
    env_cfg.events.add_base_mass = None
    env_cfg.events.randomize_rigid_body_com = None
    env_cfg.events.push_robot = None
    env_cfg.events.external_force = None
    env_cfg.events.physics_material = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)

    env_cfg.curriculum.terrain_levels = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] Environment created: num_envs={env.unwrapped.num_envs}")

    # get robot articulation handle
    robot = env.unwrapped.scene.articulations["robot"]
    foot_ids, _ = robot.find_bodies(
        name_keys=[".*_foot"],  
        preserve_order=True
    )

    # load AMP motion data
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    _MOTION_DIR = Path("datasets/motions/ecow_test") #Path("datasets/motions/rot") #Path("datasets/motions/ecow")
    
    motion_files = sorted(str(p) for p in _MOTION_DIR.glob("*.json"))
    print(f"[INFO] Found {len(motion_files)} motion files")

    # Marker 材质配置
    amp_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/AMP_Feet",
        markers={"amp_toe": sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))}
    )
    sim_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/Sim_Feet",
        markers={"sim_toe": sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))}
    )
    amp_marker = VisualizationMarkers(amp_marker_cfg)
    sim_marker = VisualizationMarkers(sim_marker_cfg)

    amp = AMPLoader(
        device=env.unwrapped.device,
        time_between_frames=env.unwrapped.step_dt,
        preload_transitions=False,
        motion_files=motion_files,
    )
    print(f"[INFO] Loaded {len(amp.trajectory_lens)} trajectories")

    # 复位环境并关闭 PD 控制（增益设为0）
    print(f"[INFO] Resetting environment...")
    env.reset()
    num_dof = robot.num_joints
    zero_gains = torch.zeros(env.unwrapped.num_envs, num_dof, device=env.unwrapped.device)
    robot.write_joint_stiffness_to_sim(zero_gains)
    robot.write_joint_damping_to_sim(zero_gains)

    # 导出目录准备
    if args_cli.export:
        os.makedirs(args_cli.output_dir, exist_ok=True)
        print(f"[INFO] Export enabled. Saving files to: {args_cli.output_dir} (Source: {args_cli.export_source})")

    actions = torch.zeros(1, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    dt = env.unwrapped.step_dt

    num_trajectories = len(amp.trajectory_lens)
    current_traj_idx = 0

    print(f"[INFO] Starting AMP trajectory replay/export...")

    while simulation_app.is_running() and current_traj_idx < num_trajectories:
        orig_file = motion_files[current_traj_idx]
        file_name = os.path.basename(orig_file)
        traj_len = float(amp.trajectory_lens[current_traj_idx])

        with open(orig_file, "r") as f:
            orig_json = json.load(f)

        recorded_frames = []
        t = 0.0

        # 计算整型步数
        num_steps = int(np.ceil(traj_len / dt)) + 1

        for step_idx in range(num_steps):
            if not simulation_app.is_running():
                break

            cur_t = min(step_idx * dt, traj_len)

            # 1. 抓取运动捕获（AMP）帧并打入仿真
            frame = amp.get_full_frame_at_time_batch(np.array([current_traj_idx]), np.array([cur_t]))

            root_pos = amp.get_root_pos_batch(frame)
            #root_pos[:, 2] =0.8 + 0.3
            root_rot = amp.get_root_rot_batch(frame)
            # root_rot[:, :3] = 0.0
            # root_rot[:, 3:4] = 1.0

            # Isaac Lab: root_pose quaternion is [w, x, y, z]; AMP data is [x, y, z, w]

            lin_vel = amp.get_linear_vel_batch(frame)
            #lin_vel[:]=0.0
            ang_vel = amp.get_angular_vel_batch(frame) #*0.0
            #ang_vel[:,0:2] = 0.0

            # 四元数与速度转换并写入物理引擎
            root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
            root_pose = torch.cat([root_pos, root_rot_wxyz], dim=-1)
            robot.write_root_pose_to_sim(root_pose)
            #root_pos[:, 2] = 0.8

            lin_vel_w = quat_apply(root_rot_wxyz, lin_vel)
            ang_vel_w = quat_apply(root_rot_wxyz, ang_vel)
            root_vel = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
            robot.write_root_velocity_to_sim(root_vel)

            joint_pos = amp.get_joint_pose_batch(frame)
            joint_vel = amp.get_joint_vel_batch(frame) 
            robot.write_joint_position_to_sim(joint_pos)
            robot.write_joint_velocity_to_sim(joint_vel)

            # 2. 推进物理仿真步
            env.step(actions)

            # 3. 足端 Marker 可视化更新
            num_feet = len(foot_ids)
            foot_pos_amp_local = amp.get_tar_toe_pos_local_batch(frame).view(-1, num_feet, 3)
            root_pos_w = robot.data.root_pos_w.unsqueeze(1)
            root_quat_w = robot.data.root_quat_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 4)

            amp_foot_pos_w = root_pos_w + math_utils.quat_apply(root_quat_w, foot_pos_amp_local.view(-1, 3)).view(-1, num_feet, 3)
            sim_foot_pos_w = robot.data.body_pos_w[:, foot_ids, :]

            amp_marker.visualize(translations=amp_foot_pos_w.view(-1, 3))
            sim_marker.visualize(translations=sim_foot_pos_w.view(-1, 3))

            # 4. 录制帧数据 (如开启导出)
            if args_cli.export:
                if args_cli.export_source == "sim":
                    # 从仿真真实物理状态提取帧
                    frame_data = extract_sim_state_as_amp_frame(robot, foot_ids, root_pos, root_rot, lin_vel, ang_vel)
                else:
                    # 使用插值后的 AMP 数据帧 (重排关节)
                    frame_np = frame.squeeze(0).cpu().numpy()
                    # 重新调整关节排列从 Isaac 到 PyBullet
                    j_pos = frame_np[7:19].reshape(3, 4).T.flatten()
                    j_vel = frame_np[37:49].reshape(3, 4).T.flatten()
                    frame_np[7:19] = j_pos
                    frame_np[37:49] = j_vel
                    frame_data = frame_np

                recorded_frames.append(frame_data.tolist())

        # 5. 轨迹循环结束，写入导出文件
        if args_cli.export and len(recorded_frames) > 0:
            export_path = os.path.join(args_cli.output_dir, file_name)
            exported_json = {
                "Loop": orig_json.get("Loop", "none"),
                "FrameDuration": float(dt),
                "MotionWeight": float(orig_json.get("MotionWeight", 1.0)),
                "Frames": recorded_frames,
            }
            with open(export_path, "w") as f:
                json.dump(exported_json, f, indent=4)
            print(f"[INFO] Successfully exported trajectory [{current_traj_idx+1}/{num_trajectories}]: "
                  f"{export_path} ({len(recorded_frames)} frames)")

        current_traj_idx += 1

    print("[INFO] Replay and Export finished.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
"""Replay AMP motion trajectories in Isaac Sim."""

import sys
import os
from isaaclab.app import AppLauncher

import argparse

# local imports
import himloco_rsl_rl.cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Replay AMP trajectories in Isaac Sim.")
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
from isaaclab.utils.math import quat_apply
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
    num_rows=10,  # number of difficulty levels
    num_cols=20,  # number of terrain types
    horizontal_scale=0.1,  # [m] grid resolution
    vertical_scale=0.005,  # [m] height scale
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        # flat地形
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    agent_cfg = himloco_rsl_rl.cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # flat plane, no terrain generator
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG #None

    # disable all domain randomization events
    env_cfg.events.add_base_mass = None
    env_cfg.events.randomize_rigid_body_com = None
    env_cfg.events.push_robot = None
    env_cfg.events.external_force = None
    env_cfg.events.physics_material = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)

    env_cfg.curriculum.terrain_levels = None

    # create environment (not wrapped — raw HimlocoManagerBasedRLEnv)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] Environment created: num_envs={env.unwrapped.num_envs}")

    # get robot articulation handle
    robot = env.unwrapped.scene.articulations["robot"]
    foot_ids,_ = robot.find_bodies(
            name_keys=[".*_foot"],  
            preserve_order=True  # 保持顺序
        )
    # load AMP motion data
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from pathlib import Path
    _MOTION_DIR = Path("datasets/motions/ecow")#Path("datasets/motions/panda") #
    print(_MOTION_DIR)
    motion_files = sorted(str(p) for p in _MOTION_DIR.glob("*.json"))
    print(f"[INFO] Found {len(motion_files)} motion files")
    # AMP 目标足端 Marker (红色小球)
    amp_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/AMP_Feet",
        markers={
            "amp_toe": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 红色
            )
        }
    )
    amp_marker = VisualizationMarkers(amp_marker_cfg)

    # Sim 实际足端 Marker (绿色小球)
    sim_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/Sim_Feet",
        markers={
            "sim_toe": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # 绿色
            )
        }
    )
    sim_marker = VisualizationMarkers(sim_marker_cfg)

    amp = AMPLoader(
        device=env.unwrapped.device,
        time_between_frames=env.unwrapped.step_dt,
        preload_transitions=False,
        motion_files=motion_files,
    )
    print(f"[INFO] Loaded {len(amp.trajectory_lens)} trajectories")

    # ---- CRITICAL: disable PD control so the robot doesn't fight the replay ----
    # In the gym replay, stiffness and damping are set to 0.
    # Without this, the PD controller will drive joints from the mocap positions
    # toward the action targets (default joint positions) during every env.step().
    # Must be called AFTER env.reset() because reset reinitializes the robot.
    print(f"[INFO] Resetting environment...")
    env.reset()
    num_dof = robot.num_joints
    zero_gains = torch.zeros(env.unwrapped.num_envs, num_dof, device=env.unwrapped.device)
    robot.write_joint_stiffness_to_sim(zero_gains)
    robot.write_joint_damping_to_sim(zero_gains)
    print(f"[INFO] Set stiffness and damping to 0 on all {num_dof} DOFs")

    # ---- Verify joint order at runtime ----
    joint_names = robot.data.joint_names
    print(f"[INFO] Asset DOF order (from PhysX articulation):")
    for i, name in enumerate(joint_names):
        print(f"  DOF[{i:2d}] = {name}")

    # ---- Verify stiffness/damping were actually zeroed ----
    actual_stiffness = robot.data.joint_stiffness
    actual_damping = robot.data.joint_damping
    print(f"[INFO] Stiffness after zeroing: min={actual_stiffness.min().item():.6f}, "
          f"max={actual_stiffness.max().item():.6f}")
    print(f"[INFO] Damping after zeroing:    min={actual_damping.min().item():.6f}, "
          f"max={actual_damping.max().item():.6f}")

    # replay loop
    traj_idx = 0
    t = 0.0
    dt = env.unwrapped.step_dt
    actions = torch.zeros(1, env.unwrapped.action_manager.total_action_dim,
                          device=env.unwrapped.device)

    print(f"[INFO] Starting AMP trajectory replay...")
    while simulation_app.is_running():
        # get mocap frame at current time
        frame = amp.get_full_frame_at_time_batch(
            np.array([traj_idx]), np.array([t]))

        root_pos = amp.get_root_pos_batch(frame)       # (1, 3)
        root_pos[:,2]+=0.3
        root_rot = amp.get_root_rot_batch(frame)       # (1, 4) in [x, y, z, w] (Isaac Gym convention)
        lin_vel = amp.get_linear_vel_batch(frame)      # (1, 3) in local body frame
        ang_vel = amp.get_angular_vel_batch(frame)     # (1, 3) in local body frame

        # Isaac Lab: root_pose quaternion is [w, x, y, z]; AMP data is [x, y, z, w]
        root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
        root_pose = torch.cat([root_pos, root_rot_wxyz], dim=-1)
        robot.write_root_pose_to_sim(root_pose)

        # Isaac Lab: root velocity is in WORLD frame; AMP data is in LOCAL body frame
        # Rotate local velocity → world velocity using [w,x,y,z] quaternion
        lin_vel_w = quat_apply(root_rot_wxyz, lin_vel)
        ang_vel_w = quat_apply(root_rot_wxyz, ang_vel)
        root_vel = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
        robot.write_root_velocity_to_sim(root_vel)

        # set joint state
        # Motion loader reorders data from AMP leg-grouped order to Isaac Lab's
        # PhysX articulation DOF order (verified at runtime):
        #   AMP data: [FL(hip,thigh,calf), FR(hip,thigh,calf), RL(hip,thigh,calf), RR(hip,thigh,calf)]
        #   Asset DOF: [hips(FL,FR,RL,RR), thighs(FL,FR,RL,RR), calves(FL,FR,RL,RR)]
        #   i.e. reshape(4,3) → transpose → flatten
        joint_pos = amp.get_joint_pose_batch(frame)    # (1, 12) in asset DOF order
        joint_vel = amp.get_joint_vel_batch(frame)     # (1, 12) in asset DOF order
        robot.write_joint_position_to_sim(joint_pos)
        #robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # ---- 足端可视化更新 ----
        num_feet = len(foot_ids)
        
        # 1. 获取 AMP 本地足端位置 (1, num_feet * 3) 并重塑为 (num_envs, num_feet, 3)
        foot_pos_amp_local = amp.get_tar_toe_pos_local_batch(frame).view(-1, num_feet, 3)

        # 2. 将 AMP 本地坐标转换为世界坐标 (P_world = P_root + R_root * P_local)
        root_pos_w = robot.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        root_quat_w = robot.data.root_quat_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 4)  # (num_envs * num_feet, 4)

        amp_foot_pos_w = root_pos_w + math_utils.quat_apply(
            root_quat_w, foot_pos_amp_local.view(-1, 3)
        ).view(-1, num_feet, 3)

        # 3. 提取 Sim 实际足端的世界坐标
        sim_foot_pos_w = robot.data.body_pos_w[:, foot_ids, :]

        # 4. 渲染 Marker (扁平化为 (N*3) 轴输入)
        amp_marker.visualize(translations=amp_foot_pos_w.view(-1, 3))
        sim_marker.visualize(translations=sim_foot_pos_w.view(-1, 3))

        # cur_footpos_translated = robot.data.body_pos_w[:, foot_ids, :] - robot.data.root_pos_w[:, :].unsqueeze(1)
        # footpos_in_body_frame = torch.zeros(env.unwrapped.num_envs, len(foot_ids), 3, device=env.unwrapped.device)
        # cur_footvel_translated = robot.data.body_lin_vel_w[:, foot_ids, :] - robot.data.root_lin_vel_w[
        #     :, :
        # ].unsqueeze(1)
        # footvel_in_body_frame = torch.zeros(env.unwrapped.num_envs, len(foot_ids), 3, device=env.unwrapped.device)
        # foot_pos = amp.get_tar_toe_pos_local_batch(frame)
        # for i in range(len(foot_ids)):
        #     footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
        #         robot.data.root_quat_w, cur_footpos_translated[:, i, :]
        #     )
        #     footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
        #         robot.data.root_quat_w, cur_footvel_translated[:, i, :]
        #     )
        #     print("i ",i," sim ",footpos_in_body_frame[:, i, :]," amp ",foot_pos[:,i*3:i*3+3])

        # step physics
        env.step(actions)

            

        t += dt
        # advance to next trajectory when current one ends
        if t + amp.time_between_frames + dt >= amp.trajectory_lens[traj_idx]:
            traj_idx = (traj_idx + 1) % len(amp.trajectory_lens)
            t = 0.0
            print(f"[INFO] Switching to trajectory {traj_idx}: "
                  f"{amp.trajectory_names[traj_idx]} "
                  f"({amp.trajectory_lens[traj_idx]:.1f}s)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

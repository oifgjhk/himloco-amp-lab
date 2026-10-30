# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
用游戏手柄或键盘控制 HimLoco RSL-RL 训练好的 RL agent 在 Isaac Sim 中交互式地执行任务。

修改日志:
- 添加 Xbox 手柄支持 (evdev), 绕过 Carb/Se2Gamepad 在 Linux 上的兼容性问题
- 手柄映射: 左摇杆→前后/左右平移, 右摇杆→Z轴旋转
- 保留 Se2Keyboard 键盘回退
- 添加 --num_envs 参数
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher # type: ignore

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive play with HimLoco RSL-RL agent (device control).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_envs", type=int, default=1, help="How many envs in simulator")

# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from collections import deque

# 切换到项目根目录，确保 logs 和 outputs 路径统一
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
import isaaclab.terrains as terrain_gen

import evdev
from evdev import InputDevice, ecodes

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, AMPOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from himloco_lab.utils import export_deploy_cfg, JointDataCollector
from himloco_lab.utils import export_himloco_policy_as_jit, export_himloco_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import himloco_lab.terrains as him_terrains

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=25.0,
    num_rows=10,  # number of difficulty levels
    num_cols=20,  # number of terrain types
    horizontal_scale=0.1,  # [m] grid resolution
    vertical_scale=0.005,  # [m] height scale
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.05, 0.50),  # 0.05 + 0.18 * difficulty
        #     step_width=0.30, 
        #     platform_width=6.0, 
        #     border_width=0.0,  
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.15,
        #     step_height_range=(0.05, 0.50),   #0.23
        #     step_width=0.30,
        #     platform_width=6.0,
        #     border_width=0.0,  
        # ),
        # "pyramid_stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.05, 0.40),  # 0.05 + 0.18 * difficulty
        #     step_width=0.25, 
        #     platform_width=3.0, 
        #     border_width=0.0,  
        # ),
        # "pyramid_stairs_inv2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.15,
        #     step_height_range=(0.05, 0.40),   #0.23
        #     step_width=0.25,
        #     platform_width=3.0,
        #     border_width=0.0,  
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, 
            slope_range=(0.0, 0.6),  
            platform_width=4.0,  
            border_width=0.0,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, 
            slope_range=(0.0, 0.6),  
            platform_width=4.0,
            border_width=0.0,  
        ),
        "hf_slope_with_noise": him_terrains.HfPyramidSlopeWithNoiseCfg(
            proportion=0.05,
            slope_range=(0.0, 0.6),
            platform_width=4.0,
            border_width=0.0,
            noise_amplitude_range=(0.01, 0.08),
            noise_step=0.005,
            downsampled_scale=0.2,
        ),
        "discrete_obstacles": him_terrains.HfDiscreteObstaclesTerrainCfg(
            proportion=0.05,
            max_height_range=(0.05, 1.2),  # 0.05 + 0.1 * difficulty
            obstacle_size_range=(2.0, 4.0),  # min=1m, max=3m
            num_obstacles=20,
            platform_width=6.0,
        ),
        # # 随机粗糙地形
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1,
        #     noise_range=(-0.02, 0.04),
        #     noise_step=0.02,
        #     border_width=0.25
        # ),
        # # 波浪地形
        # "wave": terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.1,
        #     # 波浪振幅范围（最小值，最大值），单位：米
        #     amplitude_range=(0.0, 0.2),
        #     # 波浪数量
        #     num_waves=5.0
        # ),
        # # flat地形
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.1,
        # ),
    },
)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Interactive play with HimLoco RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1] # 移除可能的 registry 前缀
    train_task_name = task_name.replace("-Play", "") # 获取对应的训练任务名称

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    # 如果 --checkpoint 是明确的文件路径（含 / 或 .pt），直接用；否则当 step 号处理
    if args_cli.checkpoint and ("/" in args_cli.checkpoint or args_cli.checkpoint.endswith(".pt")):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # 纯数字 step 号 → 拼成 "model_8000" 以匹配文件名 model_8000.pt
        ckpt = agent_cfg.load_checkpoint
        if ckpt is not None and ckpt.isdigit():
            ckpt = f"model_{ckpt}"
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, ckpt)

    if args_cli.num_envs:
        env_cfg.scene.num_envs = args_cli.num_envs

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir
    
    # Disable timeout termination for interactive play
    env_cfg.terminations.time_out = None
    #env_cfg.episode_length_s = 1.0

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs=0.0
    env_cfg.commands.base_velocity.rel_heading_envs=0.0
    env_cfg.commands.base_velocity.straight_target_prob=0.0
    env_cfg.commands.base_velocity.rotation_target_prob=0.0
    env_cfg.commands.base_velocity.lateral_target_prob=0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x=(-1.0, 1.0)
    env_cfg.scene.terrain.terrain_type = "plane"
    #env_cfg.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG
    env_cfg.curriculum.terrain_levels = None


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "interactive"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length
    )

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] history_length: {env.history_length}")
    print(f"[INFO] num_obs (total): {env.num_obs}")
    if env.num_one_step_privileged_obs is not None:
        print(f"[INFO] num_one_step_privileged_obs: {env.num_one_step_privileged_obs}")
        print(f"[INFO] privileged_history_length: {env.privileged_history_length}")
        print(f"[INFO] num_privileged_obs (total): {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    # 初始化关节数据收集器
    # collector = JointDataCollector(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # create runner from HimLoco RSL-RL
    # runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # create runner from HimLoco RSL-RL (supports HIM and AMP)
    runner_map = {"HIMOnPolicyRunner": HIMOnPolicyRunner, "AMPOnPolicyRunner": AMPOnPolicyRunner}
    runner = runner_map[agent_cfg.class_name](env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # load the checkpoint
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.device)

    # export HimLoco dual network (encoder + policy) as TorchScript JIT and ONNX
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    print(f"[INFO] Exporting HimLoco dual network to: {export_model_dir}")
    
    # Export as TorchScript JIT (combined encoder + policy)
    export_himloco_policy_as_jit(
        runner.alg.actor_critic,
        path=export_model_dir,
        policy_filename="policy.pt"
    )
    
    # Export as ONNX (separate encoder and policy)
    export_himloco_policy_as_onnx(
        runner.alg.actor_critic,
        path=export_model_dir,
        encoder_filename="encoder.onnx",
        policy_filename="policy.onnx",
        verbose=False
    )


    # Initialize control device (gamepad or keyboard)
    # Get command ranges from environment config if available
    try:
        if hasattr(env_cfg.commands, "base_velocity"):
            if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
                ranges = env_cfg.commands.base_velocity.limit_ranges
            else:
                ranges = env_cfg.commands.base_velocity.ranges
            
            max_lin_vel_x = ranges.lin_vel_x[1]
            max_lin_vel_y = ranges.lin_vel_y[1]
            max_ang_vel_z = ranges.ang_vel_z[1]
        else:
            max_lin_vel_x = 2.0
            max_lin_vel_y = 1.0
            max_ang_vel_z = 2.0
    except Exception as e:
        print(f"[WARNING] Could not read command ranges from config: {e}")
        max_lin_vel_x = 2.0
        max_lin_vel_y = 1.0
        max_ang_vel_z = 2.0
    
    # 尝试使用 evdev 直接读取 Xbox 手柄
    gamepad_device = None
    for path in evdev.list_devices():
        dev = InputDevice(path)
        if "Xbox" in dev.name or "Gamepad" in dev.name or "Joystick" in dev.name:
            gamepad_device = dev
            break

    if gamepad_device is not None:
        # 用 evdev 读取手柄，封装成和 Se2Gamepad 一样的 advance() 接口
        class EvdevGamepad:
            def __init__(self, device, v_x_sens, v_y_sens, omega_sens, dead_zone, sim_device):
                self._device = device
                self._v_x_sens = v_x_sens
                self._v_y_sens = v_y_sens
                self._omega_sens = omega_sens
                self._dead_zone = dead_zone
                self._sim_device = sim_device
                # state: lx, ly, rx, ry (normalized -1..1)
                self._state = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}

            def advance(self):
                # 非阻塞读取所有待处理事件
                while True:
                    try:
                        event = self._device.read_one()
                        if event is None:
                            break
                        # ABS_X=0, ABS_Y=1, ABS_RX=3, ABS_RY=4, ABS_Z=2, ABS_RZ=5
                        if event.type == ecodes.EV_ABS:
                            val = event.value / 32767.0  # normalize
                            if abs(val) < self._dead_zone:
                                val = 0.0
                            if event.code == ecodes.ABS_Y:      # 左摇杆上下
                                self._state["ly"] = -val  # 反号: 上推为正
                            elif event.code == ecodes.ABS_X:    # 左摇杆左右
                                self._state["lx"] = -val
                            elif event.code == ecodes.ABS_RX:   # 右摇杆左右
                                self._state["rx"] = -val
                            elif event.code == ecodes.ABS_RY:   # 右摇杆上下
                                self._state["ry"] = -val
                    except BlockingIOError:
                        break

                v_x = self._state["ly"] * self._v_x_sens
                v_y = self._state["lx"] * self._v_y_sens
                omega = self._state["rx"] * self._omega_sens
                return torch.tensor([v_x, v_y, omega], dtype=torch.float32, device=self._sim_device)

            def __str__(self):
                return f"Evdev Gamepad: {self._device.name} at {self._device.path}"

        control_device = EvdevGamepad(
            gamepad_device,
            v_x_sens=1.5,
            v_y_sens=0.8,
            omega_sens=3.14,
            dead_zone=0.1,
            sim_device=env.device,
        )
        print(f"[INFO] Using evdev gamepad: {gamepad_device.name}")
    else:
        # 回退到 Se2Gamepad (Carb 接口)
        # try:
        #     gamepad_cfg = Se2GamepadCfg(
        #         sim_device=env.device,
        #         v_x_sensitivity=max_lin_vel_x,
        #         v_y_sensitivity=max_lin_vel_y,
        #         omega_z_sensitivity=max_ang_vel_z,
        #         dead_zone=0.05,
        #     )
        #     control_device = Se2Gamepad(gamepad_cfg)
        #     print("[INFO] Using Carb gamepad control")
        # except Exception:
            keyboard_cfg = Se2KeyboardCfg(
                sim_device=env.device,
                v_x_sensitivity=max_lin_vel_x,
                v_y_sensitivity=max_lin_vel_y,
                omega_z_sensitivity=max_ang_vel_z
            )
            control_device = Se2Keyboard(keyboard_cfg)
            print("[INFO] Using keyboard control (no gamepad found)")

    print(control_device)
    print(f"[INFO] Command ranges: lin_vel_x=[{-max_lin_vel_x:.1f}, {max_lin_vel_x:.1f}], "
          f"lin_vel_y=[{-max_lin_vel_y:.1f}, {max_lin_vel_y:.1f}], "
          f"ang_vel_z=[{-max_ang_vel_z:.1f}, {max_ang_vel_z:.1f}]")

    dt = env.unwrapped.step_dt
    timestep = 0
    
    # simulate environment
    print("[INFO] Starting interactive play with real-time device control...")
    obs = env.get_observations()
    command_last = control_device.advance()
    if command_last.dim() == 1:
        command_last = command_last.unsqueeze(0)
    max_command_acc = 0.04
    
    while simulation_app.is_running():
        start_time = time.time()
        
        # Get velocity commands from control device
        command = control_device.advance()

        # Set command to environment through command manager
        try:
            vel_command_term = env.unwrapped.command_manager.get_term("base_velocity")
            if command.dim() == 1:
                command = command.unsqueeze(0)
            command_mask = torch.abs(command-command_last) <= max_command_acc
            command = command_mask * command + (~command_mask) * (command_last + torch.sign(command-command_last)*max_command_acc)
            vel_command_term.vel_command_b[:] = command.to(device=env.device)
            command_last = command
            # 每秒打印一次手柄指令
            if timestep % 50 == 0:
                print(f"[DEBUG] command: v_x={command[0,0]:.2f}, v_y={command[0,1]:.2f}, omega={command[0,2]:.2f}", end='\r')
        except Exception as e:
            print(f"[WARNING] Failed to set velocity command: {e}")
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            #obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs, _ = env.step(actions)

            # 记录关节/速度数据
            # collector.step()

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 保存图表
    # collector.plot()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

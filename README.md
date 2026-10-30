# HimLoco Lab

**HimLoco** is a reinforcement learning framework for quadruped robot locomotion, built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). It supports both standard PPO training and **AMP (Adversarial Motion Priors)** for learning natural gaits from motion capture data.

---

## Algorithms

### HimLoco (PPO)

A dual-network PPO architecture with a history-conditioned actor and a privileged critic. The actor observes proprioceptive + command + scan data; the critic additionally sees linear velocity, external forces, and privileged terrain height.

### AMP (Adversarial Motion Priors)

AMP adds a discriminator network that learns to distinguish between agent-generated and reference motion transitions. The agent receives a **style reward** from the discriminator, encouraging it to move naturally like the reference data. This produces more robust and realistic gaits compared to hand-crafted rewards alone.

```
         ┌──────────────┐
         │  Reference   │
         │  Motion Data │
         └──────┬───────┘
                │
         ┌──────▼───────┐     style reward
         │ Discriminator ├────────────────────┐
         └──────┬───────┘                    │
                │                            │
    ┌───────────┴───────────┐    ┌───────────▼───────────┐
    │   Agent Transitions   │    │    PPO Policy (AMP)    │
    └───────────────────────┘    └───────────┬───────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Task Reward +   │
                                    │  Style Reward    │
                                    └─────────────────┘
```

---

## Project Structure

```
HimLoco-AMP-Lab/
├── source/
│   ├── setup.py                         # Package setup
│   ├── pyproject.toml                   # Build system config
│   ├── config/
│   │   └── extension.toml               # Isaac Lab extension metadata
│   └── himloco_lab/
│       ├── __init__.py                  # Registers envs and UI extensions
│       ├── assets/
│       │   ├── unitree.py               # Unitree robot configs (Go2)
│       │   └── unitree_actuators.py     # Actuator models
│       ├── envs/                        # RL environment wrapper
│       ├── terrains/                    # Custom terrain generators
│       ├── tasks/locomotion/
│       │   ├── agents/
│       │   │   ├── himloco_rsl_rl_cfg.py      # Standard PPO config
│       │   │   ├── himloco_amp_rsl_rl_cfg.py  # AMP PPO config
│       │   │   └── rsl_rl_ppo_cfg.py          # Base fallback config
│       │   ├── mdp/
│       │   │   ├── commands/             # Velocity command generation
│       │   │   ├── observations.py
│       │   │   ├── rewards.py
│       │   │   ├── curriculums.py
│       │   │   └── events.py             # Domain randomization
│       │   └── robots/go2/
│       │       ├── velocity_env_cfg.py   # Standard locomotion env
│       │       └── amp_env_cfg.py        # AMP env
│       ├── rsl_rl/                       # RSL-RL training framework
│       │   ├── algorithms/               # PPO, AMP-PPO, Discriminator
│       │   ├── datasets/                 # Motion data loader
│       │   ├── modules/                  # Actor-Critic networks
│       │   ├── runners/                  # Training runners
│       │   ├── storage/                  # Rollout & replay buffers
│       │   ├── env/                      # VecEnv interface
│       │   ├── wrappers/                 # Environment wrappers
│       │   └── utils/                    # Normalizer, utilities
│       └── utils/                        # Export, plotting utilities
├── scripts/
│   ├── himloco_rsl_rl/
│   │   ├── train.py                     # Training entry point
│   │   ├── play.py                      # Policy playback
│   │   ├── play_interactive.py          # Interactive keyboard/gamepad control
│   │   ├── play_onnx.py                 # ONNX model verification
│   │   └── compare_pt_onnx.py           # PyTorch vs ONNX comparison
│   ├── list_envs.py                     # List registered environments
│   └── zero_agent.py                    # Default pose test
├── datasets/
│   └── motions/trot/                    # AMP reference motion data (*.json)
└── deploy/                              # Real-robot deployment tools
```

---

## Prerequisites

- **OS:** Linux (Ubuntu 22.04+ recommended, GLIBC >= 2.35)
- **GPU:** NVIDIA GPU with CUDA support (Isaac Sim requires CUDA)
- **Python:** 3.10 / 3.11 (3.11 recommended, matches Isaac Sim 5.x)
- **Disk:** ~30 GB free space (Isaac Sim + dependencies)

---

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n amp-lab python==3.11.0
conda activate amp-lab
```

### Step 2: Install Isaac Lab

按照 [官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) 安装 Isaac Lab。推荐使用 pip 安装方式。

### Step 3: Clone HimLoco Lab

```bash
git clone https://github.com/JF-Li5266/HimLoco-AMP-Lab.git
cd HimLoco-AMP-Lab/source
```

### Step 4: Install HimLoco Lab Package

```bash
pip install -e .
```

This installs `himloco_lab` in editable mode along with its Python dependencies (numpy, scipy, torch, tensorboard, matplotlib, etc.).

### Step 5: Download Unitree Robot Description Files

Back to the project root directory:

```bash
cd ..
```

You need the Unitree robot model files (URDF meshes or USD assets). Choose one method:

**Method 1: URDF files (recommended, Isaac Sim >= 5.0)**

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git
```

Then configure the path in `source/himloco_lab/assets/unitree.py`:

```python
UNITREE_ROS_DIR = "/home/user/projects/unitree_ros"
```

**Method 2: USD files**

```bash
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
```

Then configure the path in `source/himloco_lab/assets/unitree.py`:

```python
UNITREE_MODEL_DIR = "/home/user/projects/unitree_model"
```

**[Optional]** To switch between USD and URDF, modify `robot_cfg.spawn` in `source/himloco_lab/assets/unitree.py` (see `UNITREE_GO2_CFG` at the bottom of the file — comment/uncomment the `spawn` lines).

---

## Quick Start

### 1. Verify Installation

First, make sure HimLoco Lab is correctly installed and environments are registered:

```bash
python scripts/list_envs.py
# Expected output should include:
#   Unitree-Go2-Velocity
#   Unitree-Go2-Velocity-Play
#   Unitree-Go2-AMP
```

### 2. HimLoco Pure RL Training (Standard PPO)

Train a standard PPO policy for velocity-tracking locomotion on Unitree Go2:

```bash
# Headless mode (no GUI, faster)
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless

# With GUI (for visualization)
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity

# Custom number of environments and iterations
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless \
    --num_envs 4096 --max_iterations 10000
```

**Key parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | (required) | Registered gym environment name |
| `--agent` | `himloco_rsl_rl_cfg` | Agent config key (auto-selected per task) |
| `--num_envs` | from config | Number of parallel environments |
| `--max_iterations` | from config | Training iterations |
| `--headless` | off | Disable GUI rendering |
| `--seed` | 1 | Random seed |

**Logs and checkpoints** are saved to `logs/himloco_rsl_rl/<experiment_name>/<timestamp>/`.

### 3. HimLoco AMP Training (Adversarial Motion Priors)

Train with AMP to learn natural gaits from reference motion capture data:

```bash
# AMP training (headless)
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-AMP --headless

# With custom motion files
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-AMP --headless \
    --num_envs 4096 --max_iterations 50000
```

> **Note:** The `--agent` flag is **not needed** for AMP — the framework picks up `himloco_amp_rsl_rl_cfg` from the task's gym registration automatically.

**How AMP works:**
1. Reference motion data (`datasets/motions/trot/*.json`) is preloaded into the discriminator
2. The PPO agent generates transitions during training
3. The discriminator learns to tell apart agent vs. reference motions
4. The agent receives a **style reward** from the discriminator, encouraging natural movement

**Adding custom motion data:** Place `.json` files in `datasets/motions/<gait_name>/` and update `amp_motion_files` in [himloco_amp_rsl_rl_cfg.py](source/himloco_lab/tasks/locomotion/agents/himloco_amp_rsl_rl_cfg.py).

### 4. Playback & Evaluation

```bash
# Standard policy playback
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play

# Interactive keyboard control
python scripts/himloco_rsl_rl/play_interactive.py --task Unitree-Go2-Velocity-Play
# Controls: WASD = move, ZX = rotate, Space = reset

# ONNX model export and verification
python scripts/himloco_rsl_rl/play_onnx.py --task Unitree-Go2-Velocity-Play
```

### 5. Smoke Tests

```bash
# Random actions — verify environment runs
python scripts/random_agent.py --task Unitree-Go2-Velocity

# Zero actions — robot should stand in default pose
python scripts/zero_agent.py --task Unitree-Go2-Velocity
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'toml'` | Ensure `pyproject.toml` is in `source/` alongside `setup.py` |
| `too many values to unpack` in runner | The `step()` return count must match between env, wrapper, and runner |
| `No module named 'isaaclab'` | Isaac Lab must be installed first (see Step 2) |
| `No module named 'himloco_lab'` | Run `pip install -e .` from `source/` |
| AMP: `ValueError: 'a' cannot be empty` | Motion data not found — check `_PROJECT_ROOT` and `datasets/motions/` path |
| `psutil` version conflict | Pin `psutil==5.9.8` in setup.py (Isaac Sim 5.1 compatibility) |

---

## Configuration

### Robot Asset Paths

The Unitree model/URDF paths are configured in `source/himloco_lab/assets/unitree.py`:

| Variable | Description |
|----------|-------------|
| `UNITREE_MODEL_DIR` | Path to Unitree USD models directory |
| `UNITREE_ROS_DIR` | Path to Unitree URDF descriptions directory |

### AMP Hyperparameters

Key AMP-specific settings in [`himloco_amp_rsl_rl_cfg.py`](source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_amp_rsl_rl_cfg.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `amp_reward_coef` | 0.2 | Weight of discriminator style reward |
| `amp_replay_buffer_size` | 1,000,000 | Size of AMP experience buffer |
| `amp_discr_hidden_dims` | [1024, 512] | Discriminator architecture |
| `amp_num_preload_transitions` | 2,000,000 | Preloaded motion transitions |
| `amp_task_reward_lerp` | 0.0 | Task/style reward interpolation |

---

## Model Export

Run the playback script to export trained policies:

```bash
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play
```

Output files in the log directory:
- `policy.pt` — TorchScript JIT model (single-file deployment)
- `encoder.onnx` & `policy.onnx` — ONNX format (separated encoder/policy)

---

## Deployment

See `deploy/` for Sim2Sim (MuJoCo) validation and Sim2Real transfer tools.

---

## References

- [HimLoco](https://github.com/RoboLoco/HimLoco) — Original HimLoco implementation (IsaacGym)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — Simulation framework
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) — Unitree RL examples
- [AMP (Adversarial Motion Priors)](https://arxiv.org/abs/2104.02180) — Original AMP paper

---

## License

This project is licensed under the BSD-3-Clause License.

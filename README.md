# Trossen AI Isaac Sim

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-orange.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) [![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.0-orange.svg)](https://isaac-sim.github.io/IsaacLab) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html) [![Linux](https://img.shields.io/badge/platform-Ubuntu_22.04-lightgrey.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This repository contains NVIDIA Isaac Sim and Isaac Lab integration for Trossen AI robotic arms. It includes USD robot models, inverse kinematics-based task examples, and Isaac Lab tasks for reinforcement learning and imitation learning.

### What This Repository Offers

- Isaac Sim USD models for Trossen AI robots:
  - WidowX AI (single arm base, follower, leader left, leader right)
  - Stationary AI (dual-arm stationary platform)
  - Mobile AI (dual-arm mobile manipulator)
- Robot bringup utilities for quick model visualization and testing
- Differential inverse kinematics controller for Cartesian end-effector control
- Example scripts for pick-and-place and target following tasks
- Isaac Lab tasks for reinforcement learning (eg. reach, lift, cabinet)
- Teleoperation interface for imitation learning data collection

### Tested Environment

- Ubuntu 22.04
- Isaac Sim 5.1.0
- Isaac Lab 2.3.0
- NVIDIA GeForce RTX 5090

---

## Index

- [Overview](#overview)
- [Installation](#installation)
- [Robot Assets](#robot-assets)
- [Isaac Sim Demo Scripts](#isaac-sim-demo-scripts)
- [Isaac Lab Demo Tasks](#isaac-lab-demo-tasks)
- [Controller API](#controller-api)
- [Related Links](#related-links)

---

## Installation

### Prerequisites

Install Isaac Lab 2.3.0 following the [official installation guide](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/index.html). This will also install Isaac Sim 5.1.0. Recommended Installation Method: Binary + Source (binary download for Isaac Sim + source via git for Isaac Lab). **Note:** The recommended installation method mentioned here (Binary + Source) differs from the official recommended method.

### Clone Repository

```bash
git clone https://github.com/TrossenRobotics/trossen_ai_isaac.git
cd trossen_ai_isaac
```

### Install Trossen AI Extension (for Isaac Lab)

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e source/trossen_ai_isaac
```

Verify the environments are registered:

```bash
~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

You should see output similar to:

```
| 1  | Isaac-Open-Drawer-WXAI-v0      | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.cabinet.joint_pos_env_cfg:WXAICabinetEnvCfg           |
| 2  | Isaac-Open-Drawer-WXAI-Play-v0 | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.cabinet.joint_pos_env_cfg:WXAICabinetEnvCfg_PLAY      |
| 3  | Isaac-Lift-Cube-WXAI-v0        | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.joint_pos_env_cfg:WXAICubeLiftEnvCfg      |
| 4  | Isaac-Lift-Cube-WXAI-Play-v0   | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.joint_pos_env_cfg:WXAICubeLiftEnvCfg_PLAY |
| 5  | Isaac-Lift-Cube-WXAI-IK-Rel-v0 | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.ik_rel_env_cfg:WXAICubeLiftEnvCfg         |
| 6  | Isaac-Lift-Cube-WXAI-IK-Abs-v0 | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.ik_abs_env_cfg:WXAICubeLiftEnvCfg         |
| 7  | Isaac-Reach-WXAI-v0            | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.joint_pos_env_cfg:WXAIReachEnvCfg        |
| 8  | Isaac-Reach-WXAI-Play-v0       | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.joint_pos_env_cfg:WXAIReachEnvCfg_PLAY   |
| 9  | Isaac-Reach-WXAI-IK-Rel-v0     | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.ik_rel_env_cfg:WXAIReachEnvCfg           |
| 10 | Isaac-Reach-WXAI-IK-Abs-v0     | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.ik_abs_env_cfg:WXAIReachEnvCfg           |
```

---

## Robot Assets

All robot USD models are located in `assets/robots/`:

```
assets/robots/
├── mobile_ai/
│   └── mobile_ai.usd
├── stationary_ai/
│   └── stationary_ai.usd
└── wxai/
    ├── wxai_base.usd
    ├── wxai_follower.usd
    ├── wxai_leader_left.usd
    └── wxai_leader_right.usd
```

### Asset Generation

All USD files are generated from URDF descriptions in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description). See [assets/robots/asset_generation.md](assets/robots/asset_generation.md) for detailed generation instructions.

---

## Isaac Sim Demo Scripts

Note: Commands below assume Isaac Sim is installed at `~/isaacsim/`. Adjust the path if your installation directory differs.

### Robot Bringup

Load and visualize any robot model:

```bash
~/isaacsim/isaac-sim.sh scripts/robot_bringup.py [robot_name]
```

Supported robots: `wxai_base` (default), `wxai_follower`, `wxai_leader_left`, `wxai_leader_right`, `stationary_ai`, `mobile_ai`

### Pick and Place Demo

```bash
~/isaacsim/python.sh scripts/wxai_pick_place.py
~/isaacsim/python.sh scripts/stationary_ai_pick_place.py
~/isaacsim/python.sh scripts/mobile_ai_pick_place.py
```

### Follow Target Demo

Real-time end-effector tracking using differential IK:

```bash
~/isaacsim/python.sh scripts/wxai_follow_target.py
```

---

## Isaac Lab Demo Tasks

**Note:** Commands below assume Isaac Lab is installed at `~/IsaacLab/`. Adjust the path if your installation directory differs.

Available tasks:
- `Isaac-Reach-WXAI-v0` - Move end-effector to target pose using joint position control
- `Isaac-Reach-WXAI-IK-Rel-v0` - Reach task with relative IK delta actions
- `Isaac-Reach-WXAI-IK-Abs-v0` - Reach task with absolute IK pose actions
- `Isaac-Lift-Cube-WXAI-v0` - Pick up a cube and lift it to a target height
- `Isaac-Open-Drawer-WXAI-v0` - Open a cabinet drawer by grasping and pulling

### Reinforcement Learning

Train a policy using RSL-RL PPO:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 32 \
    --max_iterations 4000 \
    --headless
```

**Training Options:**
- `--num_envs 32`: Number of parallel environments (adjust based on GPU memory)
- `--max_iterations 4000`: Number of Iterations steps (adjust as per training tasks)
- `--headless`: Run without GUI for faster training

Training logs and checkpoints are saved to `logs/rsl_rl/<task>/<timestamp>/`.

Resume training from a checkpoint:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 32 \
    --headless \
    --resume \
    --load_run <timestamp> \
    --checkpoint <model>.pt
```

Run a trained policy:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/<task>/<timestamp>/<model>.pt
```

### Imitation Learning

Teleoperation for data collection:

```bash
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_se3_agent.py \
    --task Isaac-Reach-WXAI-IK-Rel-v0 \
    --teleop_device keyboard
```

**Teleop Device options:**
- keyboard
- spacemouse
- gamepad

---

## Controller API

The `TrossenAIController` class provides a unified interface for controlling all Trossen AI robots.

### Key Features

- Differential inverse kinematics for Cartesian end-effector control
- Gripper control with open/close commands
- Support for all robot types (WidowX AI, Stationary AI, Mobile AI)

### Basic Usage

```python
from controller import RobotType, TrossenAIController

# Initialize controller
robot = TrossenAIController(
    robot_path="/World/wxai_robot",
    robot_type=RobotType.WXAI,
    arm_dof_indices=[0, 1, 2, 3, 4, 5],
    gripper_dof_index=6,
    default_dof_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044],
)

# Command end-effector pose
robot.set_end_effector_pose(
    position=np.array([0.3, 0.0, 0.2]),
    orientation=np.array([0.7071, 0.0, 0.7071, 0.0]),  # [w, x, y, z]
)

# Gripper control
robot.open_gripper()
robot.close_gripper()

# Reset to default pose
robot.reset_to_default_pose()
```

---

## Related Links

- [Trossen Robotics](https://www.trossenrobotics.com/)
- [Trossen Arm Documentation](https://docs.trossenrobotics.com/trossen_arm/)
- [Trossen Arm Description (URDF)](https://github.com/TrossenRobotics/trossen_arm_description)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

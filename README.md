# Trossen AI Isaac Sim

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-orange.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) [![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.0-orange.svg)](https://isaac-sim.github.io/IsaacLab) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html) [![Linux](https://img.shields.io/badge/platform-Ubuntu_22.04-lightgrey.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This repository contains NVIDIA Isaac Sim assets and example scripts for Trossen AI robotic arms. It includes USD robot models, asset generation documentation, and demonstration scripts for manipulation tasks.

### What This Repository Offers

- Isaac Sim USD models for Trossen AI robots:
  - WidowX AI (single arm base, follower, leader left, leader right)
  - Stationary AI (dual-arm stationary platform)
  - Mobile AI (dual-arm mobile manipulator)
- Robot bringup utilities for quick model visualization and testing
- Differential inverse kinematics controller for Cartesian end-effector control
- Example scripts for pick-and-place and target following tasks

### Tested Environment

- Ubuntu 22.04
- Isaac Sim 5.1.0
- Python 3.11

---

## Index

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Demo Scripts](#demo-scripts)
- [Robot Assets](#robot-assets)
- [Controller API](#controller-api)
- [Related Links](#related-links)

---

## Installation

### Prerequisites

1. Install Isaac Sim 5.1.0 following the [official installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/install/install.html)

2. Verify Isaac Sim installation

   ```bash
   ~/isaacsim/isaac-sim.sh --version
   ```

### Clone Repository

```bash
cd ~
git clone https://github.com/TrossenRobotics/trossen_ai_isaac.git
cd trossen_ai_isaac
```

---

## Quick Start

> **Note:** Commands below assume Isaac Sim is installed at `~/isaacsim/`. Adjust the path if your installation directory differs.

Launch any demo script using Isaac Sim's Python interpreter:

```bash
# Visualize a robot model
~/isaacsim/isaac-sim.sh scripts/robot_bringup.py wxai_base

# Run pick-and-place demo
~/isaacsim/python.sh scripts/wxai_pick_place.py
```

---

## Demo Scripts

| Demo | Robot | Command | Description |
|------|-------|---------|-------------|
| Robot Bringup | All | `~/isaacsim/isaac-sim.sh scripts/robot_bringup.py [robot_name]` | Load and visualize robot models |
| Pick and Place | WidowX AI | `~/isaacsim/python.sh scripts/wxai_pick_place.py` | Single arm pick-and-place task |
| Follow Target | WidowX AI | `~/isaacsim/python.sh scripts/wxai_follow_target.py` | Real-time end-effector target tracking |
| Pick and Place | Stationary AI | `~/isaacsim/python.sh scripts/stationary_ai_pick_place.py` | Dual-arm coordinated handoff |
| Pick and Place | Mobile AI | `~/isaacsim/python.sh scripts/mobile_ai_pick_place.py` | Mobile base + dual-arm manipulation |

Available robot models for `robot_bringup.py`:

| Robot Name | Description |
|------------|-------------|
| `wxai_base` | Single arm base configuration (default) |
| `wxai_follower` | Single arm follower configuration |
| `wxai_leader_left` | Left leader arm for teleoperation |
| `wxai_leader_right` | Right leader arm for teleoperation |
| `stationary_ai` | Dual-arm stationary platform |
| `mobile_ai` | Dual-arm mobile manipulator |

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

---

# Trossen Arm Isaac Sim

## **Overview**

This repository provides the necessary assets and scripts for simulating robotic policies using **Trossen AI** platforms in **NVIDIA Isaac Sim**.
It includes **URDFs, mesh models, and Isaac Sim USD files** for the robot configuration, as well as Python scripts for policy execution and data collection.
Python scripts provided enable two robotic arms to **pick, transfer, and place** an object while capturing video frames and collecting robot states.

---

## Installation

To set up the Isaac Sim workspace, follow the instructions [here](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)

To install Isaac Sim python packages, follow the instructions [here](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)

Please also ensure that you have installed additional package as shown below:

```bash
pip install numpy scipy opencv-python matplotlib
```

---

## Usage

1. **Run the simulation**

Move to the cloned repository, execute the main script to start the Isaac Sim environment and begin the robotic manipulation process:

```bash
source ~/isaacsim/setup_conda_env.sh
python scripts/main.py
```

By default, it:
- Loads the Isaac Sim environment
- Randomizes the box position
- Performs pick-and-place with a handover
- Saves video footage of the simulation

2. **Modify configuration**

You can change paths and settings in `main.py`:

```bash
USD_PATH = folder_dir + "/trossen_ai_scene/trossen_ai_scene.usd"
LEFT_ARM_PATH = "/World/left_arm"
RIGHT_ARM_PATH = "/World/right_arm"
LULA_DESC_PATH = folder_dir + "/trossen_ai_scene/trossen_ai_arm.yaml"
LULA_URDF_PATH = folder_dir + "/trossen_ai_scene/wxai_follower.urdf"
```

3. **Stop the simulation**

- If `recording` in `trossen_arm_utils.py` is enabled, it will **auto-stop** when the arms reach the final position.
- Otherwise, stop manually using: **Ctrl + c**

---

## **TrossenArmController** Class

### Issac Sim API Usage

The class relies on the Isaac Sim framework, specifically:

- [**SingleArticulation**](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/articulation_controller.html) (`isaacsim.core.prims`) - Represents a robotic arm as a single articulated entity.

- [**ArticulationAction**](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/articulation_controller.html) (`isaacsim.core.utils.types`) - Defines joint movements and commands for the robotic arm.

- [**LulaTaskSpaceTrajectoryGenerator**](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/manipulators/manipulators_lula_trajectory_generator.html) (`isaacsim.robot_motion.motion_generation`) - Generates **task-space motion trajectories** for the arm.

- [**ArticulationTrajectory**](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/manipulators/manipulators_lula_trajectory_generator.html) (`isaacsim.robot_motion.motion_generation`) - Converts computed trajectories into **robotic joint actions**.

- [**ArticulationKinematicsSolver**](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_lula_kinematics.html) (`isaacsim.robot_motion.motion_generation`) - Computes **forward kinematics (FK)** for accurate arm movements.

- [**LulaKinematicsSolver**](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_lula_kinematics.html) (`isaacsim.robot_motion.motion_generation`) - Performs kinematics calculations using Lula's kinematics engine.

---

## Class Functions

### **`__init__()`**

```bash
def __init__(self, world, cameras, arm_path, solver_frame, name, lula_desc_path, lula_urdf_path, video_writer=None):
```

**Purpose**

- Loading the arm from the USD scene.
Setting up kinematics solvers for forward kinematics (FK).
- Creating a task-space trajectory planner.
- Enabling video capture (if video_writer is provided).

| Input parameter | Description |
|-------------------|---------------|
| world | The Isaac Sim world object |
| cameras | List of camera objects for recording |
| arm_path | The USD path to the robotic arm in the scene |
| solver_frame | The reference frame used for inverse kinematics |
| arm_name | The name of the robotic arm |
| lula_desc_path | Path to the robot description YAML file |
| lula_urdf_path | Path to the robot's URDF file |
| video_writer | OpenCV video writer (optional) |

---

### **`initialize()`**

```bash
def initialize(self):
```

**Purpose**

- Adds the robotic arm to the scene and initializes it.
- Moves the arm to an initial position (slightly bent).
- Records initial frames for debugging.

---

### **`slerp(start_quat, end_quat, t_values)`**

```bash
def slerp(self, start_quat, end_quat, t_values):
```

**Purpose**

- Performs Spherical Linear Interpolation (SLERP) between two quaternions for smooth rotation.
Returns a list of **interpolated** quaternions.

| Input parameter | Description |
|-------------------|---------------|
| start_quat | Starting quaternion (w, x, y, z) |
| end_quat | Target quaternion (w, x, y, z) |
| t_values | Interpolation time steps between 0 and 1 |


---

### **`move_to_target(start_pos, goal_pos, start_orientation, goal_orientation, frame)`**

```bash
def move_to_target(self, start_pos, goal_pos, start_orientation, goal_orientation, frame="ee_gripper_link"):
```

**Purpose**

- Moves the robotic arm from a starting position to a goal position, computing a smooth task-space trajectory.

| Input parameter | Description |
|-------------------|---------------|
| start_pos | Start position in XYZ coordinates |
| goal_pos | End position in XYZ coordinates |
| start_orientation | Starting orientation (quaternion) |
| goal_orientation | Target orientation (quaternion) |
| frame | Reference frame for movement (default: "ee_gripper_link") |

---

### **`apply_grasp(grasp_state, delay_steps)`**

```bash
def apply_grasp(self, grasp_state, delay_steps=100):
```

**Purpose**
- Controls the robotic gripper joints to open/close for grasping.

| Input parameter | Description |
|-------------------|---------------|
| grasp_state | Gripper joint position (0: Fully closed, 0.044: Fully open) |
| delay_steps | Number of simulation steps to hold the grasp |

---

### **`get_current_ee_position()`**

```bash
def get_current_ee_position(self):
```

**Purpose**

- Computes the current end-effector position using forward kinematics (FK).

---

### **`get_current_ee_orientation()`**

```bash
def get_current_ee_orientation(self):
```

**Purpose**

- Computes the current end-effector orientation using forward kinematics (FK).

### **`get_current_joint_velocities()`**

```bash
def get_current_joint_velocities(self):
```

**Purpose**

- Retrieves the current joints' velocity.

### **`get_current_joint_positions()`**

```bash
def get_current_joint_positions(self):
```

**Purpose**

- Retrieves the current joints' position.

### **`get_current_joint_torques()`**

```bash
def get_current_joint_torques(self):
```

**Purpose**

- Retrieves the current joint torques along the X, Y, and Z axes.

### **`get_specific_link_orientation(frame_name)`**

```bash
def get_specific_link_orientation(self, frame_name):
```

**Purpose**

- Retrieves the specific link's orientation


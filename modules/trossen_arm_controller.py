"""
Trossen Arm Controller

This module defines the `TrossenArmController` class, which manages a robotic arm in Isaac Sim. It handles 
motion planning, trajectory execution, and grasping operations.

Classes:
    - TrossenArmController: Provides methods for arm initialization, motion execution, grasping, and state retrieval.

Methods:
    - __init__(self, arm_path: str, solver_frame: str, name: str, lula_desc_path: str, lula_urdf_path: str) -> None
        Initializes the robotic arm controller.
    - initialize(self) -> None
        Initializes the arm in the simulation and moves it to the start position.
    - set_ee_pos(self, start_pos: list[float], goal_pos: list[float], start_orientation: list[float], goal_orientation: list[float], frame: str = "ee_gripper_link") -> None
        Moves the arm from a start position to a goal position using task-space motion.
    - apply_grasp(self, grasp_state: float, delay_steps: int = 100) -> None
        Controls the gripper joints to perform a grasp action.
    - get_current_ee_position(self) -> np.ndarray
        Retrieves the current end-effector position using forward kinematics.
    - get_current_ee_orientation(self) -> np.ndarray
        Retrieves the current end-effector orientation as a quaternion.
    - get_current_joint_velocities(self) -> np.ndarray
        Returns the current joint velocities of the arm.
    - get_current_joint_positions(self) -> np.ndarray
        Returns the current joint positions of the arm.
    - get_current_joint_torques(self) -> np.ndarray
        Returns the torque values applied to each joint.
    - get_specific_link_orientation(self, frame_name: str) -> np.ndarray
        Retrieves the orientation of a specific link in the robotic arm.
"""


import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import (
    LulaTaskSpaceTrajectoryGenerator, 
    ArticulationTrajectory,
    ArticulationKinematicsSolver,
    LulaKinematicsSolver
)
from isaacsim.core.utils.types import ArticulationAction

from trossen_arm_utils import *
import global_var

class TrossenArmController:
    """
    A controller class for a robotic arm in Isaac Sim, handling motion planning, trajectory execution,
    and grasping operations.
    """

    def __init__(self, arm_path: str, solver_frame: str, name: str, lula_desc_path: str, lula_urdf_path: str) -> None:
        """
        Initialize the robotic arm controller.

        :param arm_path: The USD path of the arm in the simulation.
        :type arm_path: str
        :param solver_frame: The reference frame for forward kinematics calculations.
        :type solver_frame: str
        :param name: The name of the robotic arm.
        :type name: str
        :param lula_desc_path: Path to the Lula robot description YAML file.
        :type lula_desc_path: str
        :param lula_urdf_path: Path to the Lula robot URDF file.
        :type lula_urdf_path: str
        """
        self.arm_path = arm_path
        self.arm = SingleArticulation(prim_path=arm_path, name=name)
        self.lula_urdf_path = lula_urdf_path
        self.lula_desc_path = lula_desc_path
        self.kinematics_solver = ArticulationKinematicsSolver(
            self.arm, LulaKinematicsSolver(lula_desc_path, lula_urdf_path), solver_frame
        )
        self.taskspace_generator = LulaTaskSpaceTrajectoryGenerator(lula_desc_path, lula_urdf_path)

    def initialize(self) -> None:
        """
        Initialize the robotic arm, add it to the simulation, and move it to the starting position.

        :raises RuntimeError: If the arm is not found in the simulation scene.
        """
        self.arm.initialize()
        global_var.shared_world.scene.add(self.arm)
        if not self.arm.is_valid():
            raise RuntimeError(f"Failed to find articulation at {self.arm.prim_path}")
        self.arm.apply_action(ArticulationAction(joint_positions=[0, np.pi/12, np.pi/12, 0, 0, 0, 0.044, 0.044]))
        for _ in range(60):
            capture_and_save_frames()
        self.arm_start_pos = self.get_current_ee_position()

    def set_ee_pos(self, start_pos: list[float], goal_pos: list[float], start_orientation: list[float], goal_orientation: list[float], frame: str = "ee_gripper_link") -> None:
        """
        Move the robotic arm from a start position to a goal position using task-space motion.

        :param start_pos: Start position [x, y, z].
        :type start_pos: list[float]
        :param goal_pos: Goal position [x, y, z].
        :type goal_pos: list[float]
        :param start_orientation: Start orientation as a quaternion.
        :type start_orientation: list[float]
        :param goal_orientation: Goal orientation as a quaternion.
        :type goal_orientation: list[float]
        :param frame: The reference frame for the movement (default: "ee_gripper_link").
        :type frame: str

        :raises RuntimeError: If trajectory computation fails.
        """
        num_steps = 20
        positions = np.linspace(start_pos, goal_pos, num_steps, endpoint=True)
        t_values = np.linspace(0, 1, num_steps, endpoint=True)
        orientations = slerp(start_orientation, goal_orientation, t_values)

        trajectory = self.taskspace_generator.compute_task_space_trajectory_from_points(positions, orientations, frame)

        if not trajectory:
            raise RuntimeError("Failed to compute trajectory")

        for action in ArticulationTrajectory(self.arm, trajectory, physics_dt=1/60).get_action_sequence():
            self.arm.apply_action(action)
            capture_and_save_frames()
    
    def apply_grasp(self, grasp_state: float, delay_steps: int = 100) -> None:
        """
        Apply a grasp action to control the gripper joints.

        :param grasp_state: The joint position value for grasping (0 for open, 1 for closed).
        :type grasp_state: float
        :param delay_steps: Number of simulation steps to wait after grasping (default: 100).
        :type delay_steps: int
        """
        self.arm.apply_action(ArticulationAction(joint_positions=[grasp_state, grasp_state], joint_indices=[6, 7]))
        for _ in range(delay_steps):
            capture_and_save_frames()

    def grasp(self, delay_steps: int = 100) -> None:
        """
        Close the gripper to grasp an object.

        :param delay_steps: Number of simulation steps to wait after grasping (default: 100).
        :type delay_steps: int
        """
        self.apply_grasp(0.001, delay_steps)

    def release(self, delay_steps: int = 100) -> None:
        """
        Open the gripper to release an object.

        :param delay_steps: Number of simulation steps to wait after releasing (default: 100).
        :type delay_steps: int
        """
        self.apply_grasp(0.044, delay_steps)

    def get_current_ee_position(self) -> np.ndarray:
        """
        Compute the current end-effector position using forward kinematics.

        :return: The computed position [x, y, z].
        :rtype: np.ndarray
        """
        fk_position, _ = self.kinematics_solver.compute_end_effector_pose()
        return fk_position

    def get_current_ee_orientation(self) -> np.ndarray:
        """
        Compute the current end-effector orientation using forward kinematics.

        :return: The computed orientation as a quaternion.
        :rtype: np.ndarray
        """
        _, fk_rotation_matrix = self.kinematics_solver.compute_end_effector_pose()
        rotation = R.from_matrix(fk_rotation_matrix)
        fk_orientation = rotation.as_quat()
        return fk_orientation

    def get_current_joint_velocities(self) -> np.ndarray:
        """
        Get the current joint velocities.

        :return: Velocity of each joint.
        :rtype: np.ndarray
        """
        # Get the joint velocities
        joint_velocities = self.arm.get_joint_velocities()
        return joint_velocities

    def get_current_joint_positions(self) -> np.ndarray:
        """
        Get the current joint positions.

        :return: Position of each joint.
        :rtype: np.ndarray
        """
        joint_positions = self.arm.get_joint_positions()
        return joint_positions

    
    def get_current_joint_torques(self) -> np.ndarray:
        """
        Get the current joint torques.

        :return: Torque values applied to each joint.
        :rtype: np.ndarray
        """
        measured_forces = self.arm.get_measured_joint_forces()

        # Extract torques (last three columns)
        joint_torques = measured_forces[1:, 3:]
        return joint_torques

    def get_specific_link_orientation(self, frame_name: str) -> np.ndarray:
        """
        Retrieve the orientation of a specific frame.

        :param frame_name: Name of the frame.
        :type frame_name: str

        :return: Orientation of the specified link as a quaternion.
        :rtype: np.ndarray
        """
        kinematics_solver = ArticulationKinematicsSolver(
            self.arm, LulaKinematicsSolver(self.lula_desc_path, self.lula_urdf_path), frame_name
        )
        _, fk_rotation_matrix = kinematics_solver.compute_end_effector_pose()
        rotation = R.from_matrix(fk_rotation_matrix)
        fk_orientation = rotation.as_quat()
        return fk_orientation

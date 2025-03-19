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
    - slerp(self, start_quat: list[float], end_quat: list[float], t_values: np.ndarray) -> np.ndarray
        Performs spherical linear interpolation (SLERP) between two quaternions.
    - move_to_target(self, start_pos: list[float], goal_pos: list[float], start_orientation: list[float], goal_orientation: list[float], frame: str = "ee_gripper_link") -> None
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
from scipy.spatial.transform import Rotation as R, Slerp

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

    Attributes:
        world (object): The simulation world instance.
        cameras (list): List of camera objects for capturing frames.
        arm (SingleArticulation): The robotic arm articulation.
        kinematics_solver (ArticulationKinematicsSolver): Forward kinematics solver for the arm.
        taskspace_generator (LulaTaskSpaceTrajectoryGenerator): Task-space motion planner.
        video_writer (cv2.VideoWriter, optional): Video writer for saving captured frames.
        arm_start_pos (list): Initial position of the robotic arm after initialization.
    """

    def __init__(self, arm_path: str, solver_frame: str, name: str, lula_desc_path: str, lula_urdf_path: str) -> None:
        """
        Initializes the robotic arm controller.

        Args:
            world (object): The simulation world instance.
            cameras (list): List of camera objects for capturing video frames.
            arm_path (str): The USD path of the arm in the simulation.
            solver_frame (str): The reference frame for Forward kinematics calculations.
            name (str): The name of the robotic arm.
            lula_desc_path (str): Path to the Lula robot description YAML file.
            lula_urdf_path (str): Path to the Lula robot URDF file.
            video_writer (cv2.VideoWriter, optional): Video writer for saving frames.
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
        Initializes the robotic arm, adds it to the simulation, and moves it to the starting position.

        Raises:
            RuntimeError: If the arm is not found in the simulation scene.
        """
        self.arm.initialize()
        global_var.shared_world.scene.add(self.arm)
        if not self.arm.is_valid():
            raise RuntimeError(f"Failed to find articulation at {self.arm.prim_path}")
        self.arm.apply_action(ArticulationAction(joint_positions=[0, np.pi/12, np.pi/12, 0, 0, 0, 0.044, 0.044]))
        for _ in range(60):
            capture_and_save_frames()
        self.arm_start_pos = self.get_current_ee_position()

    def slerp(self, start_quat: list[float], end_quat: list[float], t_values: np.ndarray) -> np.ndarray:
        """
        Performs spherical linear interpolation (SLERP) between two quaternions.

        Args:
            start_quat (list): Starting quaternion (w, x, y, z).
            end_quat (list): Target quaternion (w, x, y, z).
            t_values (np.ndarray): Array of interpolation time steps (0 to 1).

        Returns:
            np.ndarray: Interpolated quaternion sequence.
        """
        key_times = [0, 1]
        key_rots = R.from_quat([start_quat, end_quat])
        slerp_interpolator = Slerp(key_times, key_rots)
        interpolated_rots = slerp_interpolator(t_values)
        return interpolated_rots.as_quat()

    def move_to_target(self, start_pos: list[float], goal_pos: list[float], start_orientation: list[float], goal_orientation: list[float], frame: str = "ee_gripper_link") -> None:
        """
        Moves the robotic arm from a start position to a goal position using task-space motion.

        Args:
            start_pos (list): Start position [x, y, z].
            goal_pos (list): Goal position [x, y, z].
            start_orientation (list): Start orientation (quaternion).
            goal_orientation (list): Goal orientation (quaternion).
            frame (str): The reference frame for the movement.

        Raises:
            RuntimeError: If trajectory computation fails.
        """
        num_steps = 20
        positions = np.linspace(start_pos, goal_pos, num_steps, endpoint=True)
        t_values = np.linspace(0, 1, num_steps, endpoint=True)
        orientations = self.slerp(start_orientation, goal_orientation, t_values)

        trajectory = self.taskspace_generator.compute_task_space_trajectory_from_points(positions, orientations, frame)

        if not trajectory:
            raise RuntimeError("Failed to compute trajectory")

        for action in ArticulationTrajectory(self.arm, trajectory, physics_dt=1/60).get_action_sequence():
            self.arm.apply_action(action)
            capture_and_save_frames()
    
    def apply_grasp(self, grasp_state: float, delay_steps: int = 100) -> None:
        """
        Applies a grasp action to control the gripper joints.

        Args:
            grasp_state (float): The joint position value for grasping (0 for open, 1 for closed).
            delay_steps (int, optional): Number of simulation steps to wait after grasping.
        """
        self.arm.apply_action(ArticulationAction(joint_positions=[grasp_state, grasp_state], joint_indices=[6, 7]))
        for _ in range(delay_steps):
            capture_and_save_frames()

    def get_current_ee_position(self) -> np.ndarray:
        """
        Computes the current end-effector position using forward kinematics.

        Returns:
            - fk_position (np.ndarray): The computed position [x, y, z].
        """
        fk_position, _ = self.kinematics_solver.compute_end_effector_pose()
        return fk_position

    def get_current_ee_orientation(self) -> np.ndarray:
        """
        Computes the current end-effector orientation using forward kinematics.

        Returns:
            - fk_orientation (np.ndarray): The computed orientation as a quaternion.
        """
        _, fk_rotation_matrix = self.kinematics_solver.compute_end_effector_pose()
        rotation = R.from_matrix(fk_rotation_matrix)
        fk_orientation = rotation.as_quat()
        return fk_orientation

    def get_current_joint_velocities(self) -> np.ndarray:
        """
        Get current joints' velocities.

        Returns:
            - joint_velocities (np.ndarray): Velocity of each joint.
        """
        # Get the joint velocities
        joint_velocities = self.arm.get_joint_velocities()
        return joint_velocities

    def get_current_joint_positions(self) -> np.ndarray:
        """
        Get current joints' positions.

        Returns:
            - joint_positions (np.ndarray): Position of each joint.
        """
        joint_positions = self.arm.get_joint_positions()
        return joint_positions

    
    def get_current_joint_torques(self) -> np.ndarray:
        """
        Get current joints' torque.

        Returns:
            - joint_positions (np.ndarray): Position of each joint.
        """
        measured_forces = self.arm.get_measured_joint_forces()

        # Extract torques (last three columns)
        joint_torques = measured_forces[1:, 3:]
        return joint_torques

    def get_specific_link_orientation(self, frame_name: str) -> np.ndarray:
        """
        Get orientation of a specific frame.

        Returns:
            - fk_orientation (np.ndarray): Orientation of any link.
        """
        kinematics_solver = ArticulationKinematicsSolver(
            self.arm, LulaKinematicsSolver(self.lula_desc_path, self.lula_urdf_path), frame_name
        )
        _, fk_rotation_matrix = kinematics_solver.compute_end_effector_pose()
        rotation = R.from_matrix(fk_rotation_matrix)
        fk_orientation = rotation.as_quat()
        return fk_orientation

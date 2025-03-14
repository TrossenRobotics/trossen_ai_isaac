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

class TrossenArmController:
    def __init__(self, world, cameras, arm_path, solver_frame, name, lula_desc_path, lula_urdf_path, video_writer=None):
        self.world = world
        self.cameras = cameras
        self.arm = SingleArticulation(prim_path=arm_path, name=name)
        self.kinematics_solver = ArticulationKinematicsSolver(
            self.arm, LulaKinematicsSolver(lula_desc_path, lula_urdf_path), solver_frame
        )
        self.taskspace_generator = LulaTaskSpaceTrajectoryGenerator(lula_desc_path, lula_urdf_path)
        self.video_writer = video_writer

    def initialize(self):
        self.arm.initialize()
        self.world.scene.add(self.arm)
        if not self.arm.is_valid():
            raise RuntimeError(f"Failed to find articulation at {self.arm.prim_path}")
        self.arm.apply_action(ArticulationAction(joint_positions=[0, np.pi/12, np.pi/12, 0, 0, 0, 0.044, 0.044]))
        for _ in range(60):
            capture_and_save_frames(self.world, self.cameras, self.video_writer)
        self.arm_start_pos, _, _ = self.get_current_pos_estimation()

    def slerp(self, start_quat, end_quat, t_values):
        key_times = [0, 1]
        key_rots = R.from_quat([start_quat, end_quat])
        slerp_interpolator = Slerp(key_times, key_rots)
        interpolated_rots = slerp_interpolator(t_values)
        return interpolated_rots.as_quat()

    def move_to_target(self, start_pos, goal_pos, start_orientation, goal_orientation, frame="ee_gripper_link"):
        num_steps = 20
        positions = np.linspace(start_pos, goal_pos, num_steps, endpoint=True)
        t_values = np.linspace(0, 1, num_steps, endpoint=True)
        orientations = self.slerp(start_orientation, goal_orientation, t_values)

        trajectory = self.taskspace_generator.compute_task_space_trajectory_from_points(positions, orientations, frame)

        if not trajectory:
            raise RuntimeError("Failed to compute trajectory")

        for action in ArticulationTrajectory(self.arm, trajectory, physics_dt=1/60).get_action_sequence():
            self.arm.apply_action(action)
            capture_and_save_frames(self.world, self.cameras, self.video_writer)

    def apply_grasp(self, grasp_state, delay_steps=100):
        self.arm.apply_action(ArticulationAction(joint_positions=[grasp_state, grasp_state], joint_indices=[6, 7]))
        for _ in range(delay_steps):
            capture_and_save_frames(self.world, self.cameras, self.video_writer)

    def get_current_pos_estimation(self):
        fk_position, fk_rotation_matrix = self.kinematics_solver.compute_end_effector_pose()
        rotation = R.from_matrix(fk_rotation_matrix)
        fk_orientation = rotation.as_quat()
        yaw, _ = quaternion_to_yaw(fk_orientation)
        return fk_position, fk_orientation, yaw

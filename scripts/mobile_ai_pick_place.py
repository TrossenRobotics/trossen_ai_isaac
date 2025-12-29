# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Mobile AI Sequential Pick-and-Place Demonstration.

This script demonstrates mobile base motion and dual-arm pick-and-place manipulation using the Mobile AI robot.

Usage:
    ~/isaacsim/python.sh scripts/mobile_ai_pick_place.py
"""

from __future__ import annotations

import os
import sys

from isaacsim import SimulationApp

# Must initialize SimulationApp before importing other Isaac Sim modules
simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial  # noqa: E402
from isaacsim.core.experimental.objects import Cube  # noqa: E402
from isaacsim.core.experimental.prims import (  # noqa: E402
    GeomPrim,
    RigidPrim,
    XformPrim,
)
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController  # noqa: E402

# Default configuration constants
DEFAULT_CUBE_SIZE = np.array([0.05, 0.05, 0.05])
DEFAULT_CUBE_POSITION = np.array([0.3, 0.5, 0.05])
DEFAULT_CUBE_ORIENTATION = np.array([1, 0, 0, 0])
DEFAULT_INTERMEDIATE_POSITION = np.array([0.0, 0.5, 0.05])
DEFAULT_TARGET_POSITION = np.array([-0.3, 0.5, 0.05])
LEFT_ARM_HOME_POSITION = np.array([0.3, 0.5, 0.4])
RIGHT_ARM_HOME_POSITION = np.array([-0.3, 0.5, 0.4])

# Phase timing (frames) for pick-place sequences
LEFT_ARM_EVENTS_DT = [80, 50, 10, 50, 80, 50, 10, 50, 80]
RIGHT_ARM_EVENTS_DT = [80, 50, 10, 50, 80, 50, 10, 50, 80]

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15
APPROACH_OFFSET = np.array([0.0, 0.0, 0.03])
DOWNWARD_ORIENTATION = np.array(
    [0.5, 0.5, 0.5, -0.5]
)  # Gripper orientation [w, x, y, z]

# Scene configuration
ROBOT_USD_PATH = "./assets/robots/mobile_ai/mobile_ai.usd"
ROBOT_SCENE_PATH = "/World/mobile_ai"
ROBOT_SPAWN_POSITION = np.array([0.0, 2.1, -0.70])
ROBOT_ORIENTATION = np.array(
    [0.7071068, 0.0, 0.0, -0.7071068]
)  # -90° around Z [w, x, y, z]
ENVIRONMENT_SCENE_PATH = "/World/Environment"
CUBE_SCENE_PATH = "/World/Cube"
CAMERA_EYE_POSITION = [-3.0, 3.0, 2.5]
CAMERA_TARGET_POSITION = [0.0, 0.5, 0.0]

# Mobile base driving configuration
WHEEL_LEFT_DOF_INDEX = 4
WHEEL_RIGHT_DOF_INDEX = 5
DRIVE_VELOCITY = 90
DRIVE_DURATION_STEPS = 200

# Robot controller configuration
LEFT_ARM_DOF_INDICES = [10, 12, 14, 16, 18, 20]
LEFT_GRIPPER_DOF_INDEX = 22
RIGHT_ARM_DOF_INDICES = [11, 13, 15, 17, 19, 21]
RIGHT_GRIPPER_DOF_INDEX = 24
MOBILE_AI_DEFAULT_DOF_POSITIONS = (
    [0.0] * 10 + [0.0, 0.0] * 6 + [0.044, 0.044, 0.044, 0.044]
)
STOP_VELOCITY_TARGETS = [0.0] * 26
DRIVE_FORWARD_VELOCITY_TARGETS = (
    [0.0] * 4 + [DRIVE_VELOCITY, DRIVE_VELOCITY] + [0.0] * 20
)


class MobileAIPickPlace:
    """Sequential dual-arm pick-and-place task for Mobile AI robot."""

    def __init__(
        self,
        left_events_dt: list[int] | None = None,
        right_events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        cube_size: np.ndarray | None = None,
        intermediate_position: np.ndarray | None = None,
    ):
        """Initialize sequential dual-arm pick-and-place task.

        Args:
            left_events_dt: Time deltas for left arm pick-place sequence.
            right_events_dt: Time deltas for right arm pick-place sequence.
            cube_initial_position: Initial cube position [x, y, z] in meters.
            cube_initial_orientation: Initial cube orientation quaternion [w, x, y, z].
            cube_size: Size of the cube [width, height, depth] in meters.
            intermediate_position: Position where left arm places / right arm picks.
        """
        self.cube_size = cube_size if cube_size is not None else DEFAULT_CUBE_SIZE
        self.cube_initial_position = (
            cube_initial_position
            if cube_initial_position is not None
            else DEFAULT_CUBE_POSITION
        )
        self.cube_initial_orientation = (
            cube_initial_orientation
            if cube_initial_orientation is not None
            else DEFAULT_CUBE_ORIENTATION
        )
        self.intermediate_position = (
            intermediate_position
            if intermediate_position is not None
            else DEFAULT_INTERMEDIATE_POSITION
        )

        self.left_events_dt = (
            left_events_dt if left_events_dt is not None else LEFT_ARM_EVENTS_DT
        )
        self.right_events_dt = (
            right_events_dt if right_events_dt is not None else RIGHT_ARM_EVENTS_DT
        )

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET
        self.left_home_position = LEFT_ARM_HOME_POSITION
        self.right_home_position = RIGHT_ARM_HOME_POSITION
        self.downward_orientation = DOWNWARD_ORIENTATION
        self.target_position = DEFAULT_TARGET_POSITION

        self.cube = None
        self.robot_left = None
        self.robot_right = None

        # Trajectory state
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_index = 0
        self.current_phase = (
            "driving"  # "driving", "left_arm_pickup", or "right_arm_pickup"
        )
        self.left_total_steps = sum(self.left_events_dt)
        self.right_total_steps = sum(self.right_events_dt)
        self.drive_step_counter = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        stage_utils.create_new_stage(template="sunlight")

        # Add Simple Room environment with table
        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Simple_Room/simple_room.usd",
            path=ENVIRONMENT_SCENE_PATH,
        )

        # Spawn robot in scene at specified position and orientation
        stage_utils.add_reference_to_stage(
            usd_path=ROBOT_USD_PATH,
            path=ROBOT_SCENE_PATH,
        )

        robot_xform = XformPrim(ROBOT_SCENE_PATH)
        robot_xform.set_world_poses(
            positions=ROBOT_SPAWN_POSITION,
            orientations=ROBOT_ORIENTATION,
        )

        # Create left arm controller
        self.robot_left = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.MOBILE_AI,
            end_effector_link=RigidPrim(f"{ROBOT_SCENE_PATH}/follower_left_link_6"),
            arm_dof_indices=LEFT_ARM_DOF_INDICES,
            gripper_dof_index=LEFT_GRIPPER_DOF_INDEX,
            default_dof_positions=MOBILE_AI_DEFAULT_DOF_POSITIONS,
        )

        # Create right arm controller
        self.robot_right = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.MOBILE_AI,
            end_effector_link=RigidPrim(f"{ROBOT_SCENE_PATH}/follower_right_link_6"),
            arm_dof_indices=RIGHT_ARM_DOF_INDICES,
            gripper_dof_index=RIGHT_GRIPPER_DOF_INDEX,
            default_dof_positions=MOBILE_AI_DEFAULT_DOF_POSITIONS,
        )

        visual_material = PreviewSurfaceMaterial("/Visual_materials/red")
        visual_material.set_input_values("diffuseColor", [1.0, 0.0, 0.0])

        cube_shape = Cube(
            paths=CUBE_SCENE_PATH,
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=[1.0],
            scales=self.cube_size,
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

        set_camera_view(eye=CAMERA_EYE_POSITION, target=CAMERA_TARGET_POSITION)

    def forward(self) -> bool:
        """Execute one simulation step of the sequential pick-and-place.

        Returns:
            bool: True if task is in progress, False if complete.
        """
        if self.is_done():
            return False

        # Driving phase: Move mobile base forward to manipulation position
        if self.current_phase == "driving":
            if self.drive_step_counter < DRIVE_DURATION_STEPS:
                self.robot_left.set_dof_velocity_targets(DRIVE_FORWARD_VELOCITY_TARGETS)
                self.drive_step_counter += 1

            else:
                print("Driving complete. Starting manipulation...")
                self.robot_left.set_dof_velocity_targets(STOP_VELOCITY_TARGETS)
                self.current_phase = "left_arm_pickup"

            return True

        if self.left_trajectory is None:
            self.generate_left_arm_trajectory()
        if self.right_trajectory is None:
            self.generate_right_arm_trajectory()

        if self.current_phase == "left_arm_pickup":
            if self.trajectory_index < len(self.left_trajectory):
                left_pos, left_ori, _ = self.left_trajectory[self.trajectory_index]
                self.robot_left.set_end_effector_pose(
                    position=left_pos.reshape(1, -1),
                    orientation=left_ori.reshape(1, -1),
                )

                phase_boundaries = [0]
                cumulative = 0
                for duration in self.left_events_dt:
                    cumulative += duration
                    phase_boundaries.append(cumulative)

                if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                    self.robot_left.close_gripper()
                elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                    self.robot_left.open_gripper()

                self.trajectory_index += 1

                if self.trajectory_index >= len(self.left_trajectory):
                    print("Left arm pick-place complete. Starting right arm...")
                    self.current_phase = "right_arm_pickup"
                    self.trajectory_index = 0

        elif self.current_phase == "right_arm_pickup":
            if self.trajectory_index < len(self.right_trajectory):
                right_pos, right_ori, _ = self.right_trajectory[self.trajectory_index]
                self.robot_right.set_end_effector_pose(
                    position=right_pos.reshape(1, -1),
                    orientation=right_ori.reshape(1, -1),
                )

                phase_boundaries = [0]
                cumulative = 0
                for duration in self.right_events_dt:
                    cumulative += duration
                    phase_boundaries.append(cumulative)

                if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                    self.robot_right.close_gripper()
                elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                    self.robot_right.open_gripper()

                self.trajectory_index += 1

        return True

    def generate_left_arm_trajectory(self) -> None:
        """Generate pick-place trajectory for left arm.

        Left arm picks from cube_initial_position and places at intermediate_position.
        Creates a 9-phase pick-and-place trajectory:
        Phase 0: Move to pre-pick position above cube
        Phase 1: Descend to approach position
        Phase 2: Grasp cube
        Phase 3: Lift cube with clearance
        Phase 4: Move to pre-place position
        Phase 5: Descend to place approach position
        Phase 6: Release cube
        Phase 7: Lift to clearance height
        Phase 8: Return to home position
        """
        cube_pos = self.cube.get_world_poses()[0].numpy().flatten()
        place_pos = self.intermediate_position

        _, current_ee_pos, _ = self.robot_left.get_current_state()
        current_ee_pos = current_ee_pos[0]

        key_frames = [
            current_ee_pos,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            cube_pos + self.approach_offset,
            cube_pos + self.approach_offset,  # Grasp
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + self.approach_offset,
            place_pos + self.approach_offset,  # Release
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.left_home_position,
        ]

        orientations = [self.downward_orientation for _ in key_frames]

        self.left_trajectory = self.make_trajectory(
            key_frames, orientations, self.left_events_dt
        )

    def generate_right_arm_trajectory(self) -> None:
        """Generate pick-place trajectory for right arm.

        Right arm picks from intermediate_position and places at target_position.
        Creates a 9-phase pick-and-place trajectory:
        Phase 0: Move to pre-pick position above intermediate position
        Phase 1: Descend to approach position
        Phase 2: Grasp cube
        Phase 3: Lift cube with clearance
        Phase 4: Move to pre-place position above target
        Phase 5: Descend to place approach position
        Phase 6: Release cube
        Phase 7: Lift to clearance height
        Phase 8: Return to home position
        """
        pick_pos = self.intermediate_position
        place_pos = self.target_position

        _, current_ee_pos, _ = self.robot_right.get_current_state()
        current_ee_pos = current_ee_pos[0]

        key_frames = [
            current_ee_pos,
            pick_pos + np.array([0.0, 0.0, self.clearance_height]),
            pick_pos + self.approach_offset,
            pick_pos + self.approach_offset,  # Grasp
            pick_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + self.approach_offset,
            place_pos + self.approach_offset,  # Release
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.right_home_position,
        ]

        orientations = [self.downward_orientation for _ in key_frames]

        self.right_trajectory = self.make_trajectory(
            key_frames, orientations, self.right_events_dt
        )

    def make_trajectory(
        self,
        key_frames: list[np.ndarray],
        orientations: list[np.ndarray],
        dt: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Generate smooth trajectory via linear interpolation between keyframes.

        Args:
            key_frames: Position waypoints [x, y, z] in meters. Length must be len(dt) + 1.
            orientations: Orientation quaternions [w, x, y, z] for each keyframe.
            dt: Duration in steps for each trajectory segment.

        Returns:
            List of (position, orientation, cumulative_step) tuples.
        """
        if len(key_frames) != len(dt) + 1:
            raise ValueError(f"Expected {len(dt) + 1} keyframes for {len(dt)} segments")
        if len(orientations) != len(key_frames):
            raise ValueError("Orientations must match keyframe count")

        trajectory = []
        cumulative_step = 0

        for i in range(len(dt)):
            start_pos = np.array(key_frames[i], dtype=np.float64)
            end_pos = np.array(key_frames[i + 1], dtype=np.float64)
            start_ori = np.array(orientations[i], dtype=np.float64)
            n_steps = dt[i]

            for step in range(n_steps):
                alpha = step / n_steps if n_steps > 0 else 0.0
                interpolated_pos = start_pos + alpha * (end_pos - start_pos)
                trajectory.append((interpolated_pos, start_ori, cumulative_step + step))

            cumulative_step += n_steps

        trajectory.append(
            (
                np.array(key_frames[-1], dtype=np.float64),
                np.array(orientations[-1], dtype=np.float64),
                cumulative_step,
            )
        )

        return trajectory

    def is_done(self) -> bool:
        """Check if task is complete.

        Returns:
            bool: True if both driving and arm sequences are complete.
        """
        if self.current_phase == "driving" or self.current_phase == "left_arm_pickup":
            return False
        return self.right_trajectory is not None and self.trajectory_index >= len(
            self.right_trajectory
        )

    def reset(
        self,
        cube_position: np.ndarray | None = None,
        cube_orientation: np.ndarray | None = None,
    ) -> None:
        """Reset task to initial state."""
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)

    def reset_robot(self) -> None:
        """Reset both arms to default pose and clear trajectories."""
        if self.robot_left is None:
            raise RuntimeError("Cannot reset robot: left arm not initialized.")
        if self.robot_right is None:
            raise RuntimeError("Cannot reset robot: right arm not initialized.")

        self.robot_left.reset_to_default_pose()
        self.robot_left.open_gripper()
        self.robot_right.reset_to_default_pose()
        self.robot_right.open_gripper()
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_index = 0
        self.current_phase = "driving"
        self.drive_step_counter = 0

    def reset_cube(
        self, position: np.ndarray | None = None, orientation: np.ndarray | None = None
    ) -> None:
        """Reset cube to specified or initial pose."""
        if self.cube is None:
            raise RuntimeError("Cannot reset cube: cube not initialized.")

        reset_position = (
            position if position is not None else self.cube_initial_position
        )
        reset_orientation = (
            orientation if orientation is not None else self.cube_initial_orientation
        )
        self.cube.set_world_poses(
            positions=reset_position.reshape(1, -1),
            orientations=reset_orientation.reshape(1, -1),
        )


def main():
    print("Mobile AI Sequential Dual-Arm Pick-and-Place Demo")
    print("Demo sequence:")
    print("  1. Mobile base navigates forward to manipulation area")
    print("  2. Left arm picks cube and places at intermediate position")
    print(
        "  3. Right arm picks cube from intermediate position and places at final target"
    )
    simulation_app.update()

    pick_place = MobileAIPickPlace()
    pick_place.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    task_completed = False
    pick_place.reset()

    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            pick_place.forward()

        if pick_place.is_done() and not task_completed:
            print("Task complete - cube moved from left to right!")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping mobile AI pick and place demo...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()

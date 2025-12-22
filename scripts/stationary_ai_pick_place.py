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
Stationary AI Pick-and-Place Demonstration.

This script demonstrates dual-arm pick-and-place manipulation task using the Stationary AI robot.

Usage:
    ~/isaacsim/python.sh scripts/stationary_ai_pick_place.py
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
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController  # noqa: E402

# Default configuration constants
DEFAULT_CUBE_SIZE = np.array([0.05, 0.05, 0.05])
DEFAULT_CUBE_POSITION = np.array([0.0, 0.25, 0.06])
DEFAULT_CUBE_ORIENTATION = np.array([1, 0, 0, 0])
DEFAULT_TARGET_POSITION = np.array([0.0, -0.25, 0.06])
LEFT_ARM_HOME_POSITION = np.array([0.0, 0.3, 0.3])
RIGHT_ARM_HOME_POSITION = np.array([0.0, -0.3, 0.3])
DEFAULT_EVENTS_DT = [40, 40, 40, 20, 40, 80, 40, 20, 20, 50, 50, 60, 40, 20, 50]

# Trajectory parameters
HANDOFF_OFFSET = 0.04  # Distance each gripper is from center during handoff in meters
CENTER_HANDOFF_POSITION = np.array(
    [0.0, 0.0, 0.25]
)  # Center position for dual-arm handoff
CLEARANCE_HEIGHT = 0.15  # Height clearance above objects in meters
APPROACH_OFFSET = np.array([0.0, 0.0, 0.03])  # Vertical offset for approach in meters


# Gripper orientations (quaternion [w, x, y, z])
LEFT_ARM_DOWNWARD_ORIENTATION = np.array(
    [[0.5, 0.5, 0.5, -0.5]]
)  # Gripper pointing down
RIGHT_ARM_DOWNWARD_ORIENTATION = np.array(
    [[0.5, -0.5, 0.5, 0.5]]
)  # Gripper pointing down
LEFT_ARM_HANDOFF_ORIENTATION = np.array(
    [[0.7071068, 0.0, 0.0, -0.7071068]]
)  # Gripper pointing toward right arm
RIGHT_ARM_HANDOFF_ORIENTATION = np.array(
    [[0.7071068, 0.0, 0.0, 0.7071068]]
)  # Gripper pointing toward left arm
RIGHT_ARM_RECEIVE_ORIENTATION = np.array(
    [[0.5, 0.5, 0.5, 0.5]]
)  # Rotated to avoid finger collision

# Scene configuration
ROBOT_USD_PATH = "./assets/robots/stationary_ai/stationary_ai.usd"
ROBOT_SCENE_PATH = "/World/stationary_ai"
GROUND_SCENE_PATH = "/World/ground"
CUBE_SCENE_PATH = "/World/Cube"

# Robot controller configuration
LEFT_ARM_DOF_INDICES = [0, 2, 4, 6, 8, 10]
LEFT_GRIPPER_DOF_INDEX = 12
RIGHT_ARM_DOF_INDICES = [1, 3, 5, 7, 9, 11]
RIGHT_GRIPPER_DOF_INDEX = 14
STATIONARY_AI_DEFAULT_DOF_POSITIONS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.044,
    0.044,
    0.044,
    0.044,
]


class StationaryAIPickPlace:
    """Dual-arm pick-and-place task with coordinated handoff."""

    def __init__(
        self,
        events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        cube_size: np.ndarray | None = None,
        pick_position: np.ndarray | None = None,
        place_position: np.ndarray | None = None,
        handoff_position: np.ndarray | None = None,
    ):
        """Initialize dual-arm pick-and-place task.

        Args:
            events_dt: List of time deltas for events in the task sequence.
            cube_initial_position: Initial position [x, y, z] of the cube in meters.
            cube_initial_orientation: Initial orientation quaternion [w, x, y, z] of the cube.
            cube_size: Size of the cube [width, height, depth] in meters.
            pick_position: Target position [x, y, z] for picking the cube in meters.
            place_position: Target position [x, y, z] for placing the cube in meters.
            handoff_position: Position [x, y, z] for handoff between robots in meters.
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
            else np.array([1, 0, 0, 0])
        )
        self.pick_position = (
            pick_position if pick_position is not None else DEFAULT_CUBE_POSITION
        )
        self.place_position = (
            place_position if place_position is not None else DEFAULT_TARGET_POSITION
        )
        self.handoff_position = (
            handoff_position
            if handoff_position is not None
            else CENTER_HANDOFF_POSITION
        )

        self.events_dt = events_dt if events_dt is not None else DEFAULT_EVENTS_DT

        self.handoff_offset = HANDOFF_OFFSET
        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET
        self.left_home_position = LEFT_ARM_HOME_POSITION
        self.right_home_position = RIGHT_ARM_HOME_POSITION

        self.cube = None
        self.robot_left = None
        self.robot_right = None
        self.trajectory_left = None
        self.trajectory_right = None
        self.trajectory_index = 0
        self.phase_boundaries = None
        self.total_steps = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        stage_utils.create_new_stage(template="sunlight")

        # Spawn robot in scene
        stage_utils.add_reference_to_stage(
            usd_path=ROBOT_USD_PATH,
            path=ROBOT_SCENE_PATH,
        )

        self.robot_left = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.STATIONARY_AI,
            end_effector_link=RigidPrim(f"{ROBOT_SCENE_PATH}/follower_left_link_6"),
            arm_dof_indices=LEFT_ARM_DOF_INDICES,
            gripper_dof_index=LEFT_GRIPPER_DOF_INDEX,
            default_dof_positions=STATIONARY_AI_DEFAULT_DOF_POSITIONS,
        )
        self.robot_right = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.STATIONARY_AI,
            end_effector_link=RigidPrim(f"{ROBOT_SCENE_PATH}/follower_right_link_6"),
            arm_dof_indices=RIGHT_ARM_DOF_INDICES,
            gripper_dof_index=RIGHT_GRIPPER_DOF_INDEX,
            default_dof_positions=STATIONARY_AI_DEFAULT_DOF_POSITIONS,
        )

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Grid/default_environment.usd",
            path=GROUND_SCENE_PATH,
        )

        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

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

    def forward(self) -> bool:
        """Execute one simulation step of the dual-arm handoff sequence.

        Returns:
            bool: True if sequence is in progress, False if complete.
        """
        if self.is_done():
            return False

        if self.trajectory_left is None or self.trajectory_right is None:
            self.generate_dual_arm_trajectory()

        if self.trajectory_index < len(self.trajectory_left):
            left_pos, left_ori, _ = self.trajectory_left[self.trajectory_index]
            self.robot_left.set_end_effector_pose(
                position=left_pos.reshape(1, -1),
                orientation=left_ori.reshape(1, -1),
            )

            right_pos, right_ori, _ = self.trajectory_right[self.trajectory_index]
            self.robot_right.set_end_effector_pose(
                position=right_pos.reshape(1, -1),
                orientation=right_ori.reshape(1, -1),
            )

            self.trajectory_index += 1

            phase_boundaries = [0]
            cumulative = 0
            for duration in self.events_dt:
                cumulative += duration
                phase_boundaries.append(cumulative)

            if phase_boundaries[3] <= self.trajectory_index < phase_boundaries[4]:
                self.robot_left.close_gripper()
            elif phase_boundaries[8] <= self.trajectory_index < phase_boundaries[9]:
                self.robot_left.open_gripper()

            if phase_boundaries[7] <= self.trajectory_index < phase_boundaries[8]:
                self.robot_right.close_gripper()
            elif phase_boundaries[13] <= self.trajectory_index < phase_boundaries[14]:
                self.robot_right.open_gripper()

        return True

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

        Raises:
            ValueError: If array lengths are incompatible.
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
            end_ori = np.array(orientations[i + 1], dtype=np.float64)
            n_steps = dt[i]

            # Linear interpolation for each step in this segment
            for step in range(n_steps):
                alpha = step / n_steps if n_steps > 0 else 0.0
                interpolated_pos = start_pos + alpha * (end_pos - start_pos)
                # Interpolate orientation
                interpolated_ori = start_ori + alpha * (end_ori - start_ori)
                interpolated_ori = interpolated_ori / np.linalg.norm(interpolated_ori)
                trajectory.append(
                    (interpolated_pos, interpolated_ori, cumulative_step + step)
                )

            cumulative_step += n_steps

        trajectory.append(
            (
                np.array(key_frames[-1], dtype=np.float64),
                np.array(orientations[-1], dtype=np.float64),
                cumulative_step,
            )
        )

        return trajectory

    def generate_dual_arm_trajectory(self) -> None:
        """Generate complete dual-arm handoff trajectory.

        Creates a 15-phase coordinated trajectory for both arms:
        Phase 0: Right arm moves to wait position, left prepares
        Phase 1: Left descends to pre-pick
        Phase 2: Left descends to pick
        Phase 3: Left grasps
        Phase 4: Left lifts with cube
        Phase 5: Left moves to handoff, right moves to pre-handoff
        Phase 6: Right moves closer to receive
        Phase 7: Right grasps cube
        Phase 8: Left releases
        Phase 9: Left retreats
        Phase 10: Right lifts and rotates
        Phase 11: Right moves to pre-place
        Phase 12: Right descends to place
        Phase 13: Right releases
        Phase 14: Right retreats to home
        """
        cube_pos = self.pick_position

        left_home = self.left_home_position
        left_prepick = cube_pos + np.array([0.0, 0.0, self.clearance_height])
        left_pick = cube_pos + self.approach_offset
        left_handoff = self.handoff_position + np.array([0.0, self.handoff_offset, 0.0])
        left_retreat = self.left_home_position

        left_key_frames = [
            left_home,
            left_prepick,
            left_prepick,
            left_pick,
            left_pick,
            left_prepick,
            left_handoff,
            left_handoff,
            left_handoff,
            left_handoff,
            left_retreat,
            left_retreat,
            left_retreat,
            left_retreat,
            left_retreat,
            left_retreat,
        ]

        left_orientations = [
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_HANDOFF_ORIENTATION[0],
            LEFT_ARM_HANDOFF_ORIENTATION[0],
            LEFT_ARM_HANDOFF_ORIENTATION[0],
            LEFT_ARM_HANDOFF_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
            LEFT_ARM_DOWNWARD_ORIENTATION[0],
        ]

        right_home = self.right_home_position
        right_wait = np.array([0.0, -0.20, 0.25])
        right_pre_handoff = self.handoff_position + np.array([0.0, -0.12, 0.0])
        right_handoff = self.handoff_position + np.array(
            [0.0, -self.handoff_offset, 0.0]
        )
        lifted_handoff = right_handoff + np.array([0.0, 0.0, 0.05])
        right_preplace = self.place_position + np.array(
            [0.0, 0.0, self.clearance_height]
        )
        right_place = self.place_position + self.approach_offset

        right_key_frames = [
            right_home,
            right_wait,
            right_wait,
            right_wait,
            right_wait,
            right_wait,
            right_pre_handoff,
            right_handoff,
            right_handoff,
            right_handoff,
            right_handoff,
            lifted_handoff,
            right_preplace,
            right_place,
            right_place,
            right_home,
        ]

        right_orientations = [
            RIGHT_ARM_DOWNWARD_ORIENTATION[0],
            RIGHT_ARM_HANDOFF_ORIENTATION[0],
            RIGHT_ARM_HANDOFF_ORIENTATION[0],
            RIGHT_ARM_HANDOFF_ORIENTATION[0],
            RIGHT_ARM_HANDOFF_ORIENTATION[0],
            RIGHT_ARM_HANDOFF_ORIENTATION[0],
            RIGHT_ARM_RECEIVE_ORIENTATION[0],
            RIGHT_ARM_RECEIVE_ORIENTATION[0],
            RIGHT_ARM_RECEIVE_ORIENTATION[0],
            RIGHT_ARM_RECEIVE_ORIENTATION[0],
            RIGHT_ARM_RECEIVE_ORIENTATION[0],
            RIGHT_ARM_HANDOFF_ORIENTATION[0],
            RIGHT_ARM_DOWNWARD_ORIENTATION[0],
            RIGHT_ARM_DOWNWARD_ORIENTATION[0],
            RIGHT_ARM_DOWNWARD_ORIENTATION[0],
            RIGHT_ARM_DOWNWARD_ORIENTATION[0],
        ]

        self.trajectory_left = self.make_trajectory(
            left_key_frames, left_orientations, self.events_dt
        )
        self.trajectory_right = self.make_trajectory(
            right_key_frames, right_orientations, self.events_dt
        )
        self.trajectory_index = 0

    def is_done(self) -> bool:
        """Check if dual-arm handoff task is complete.

        Returns:
            bool: True if all phases have been executed.
        """
        return self.trajectory_left is not None and self.trajectory_index >= len(
            self.trajectory_left
        )

    def reset(
        self,
        cube_position: np.ndarray | None = None,
        cube_orientation: np.ndarray | None = None,
    ) -> None:
        """Reset task to initial state.

        Args:
            cube_position: Optional cube position override.
            cube_orientation: Optional cube orientation override.
        """
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)

    def reset_robot(self) -> None:
        """Reset both arms to default pose and clear phase tracking."""
        if self.robot_left is not None:
            self.robot_left.reset_to_default_pose()
            self.robot_left.open_gripper()
        if self.robot_right is not None:
            self.robot_right.reset_to_default_pose()
            self.robot_right.open_gripper()
        self.trajectory_left = None
        self.trajectory_right = None
        self.trajectory_index = 0

    def reset_cube(
        self, position: np.ndarray | None = None, orientation: np.ndarray | None = None
    ) -> None:
        """Reset cube to specified or initial pose."""
        if self.cube is not None:
            reset_position = (
                position if position is not None else self.cube_initial_position
            )
            reset_orientation = (
                orientation
                if orientation is not None
                else self.cube_initial_orientation
            )
            self.cube.set_world_poses(
                positions=reset_position.reshape(1, -1),
                orientations=reset_orientation.reshape(1, -1),
            )


def main():
    print("Stationary AI Dual-Arm Pick-and-Place Demo")
    simulation_app.update()

    pick_place = StationaryAIPickPlace()
    pick_place.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    reset_needed = True
    task_completed = False

    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            if reset_needed:
                pick_place.reset()
                reset_needed = False

            pick_place.forward()

        if pick_place.is_done() and not task_completed:
            print("Task complete")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping dual-arm pick and place demo...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()

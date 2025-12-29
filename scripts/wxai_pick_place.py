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
WidowX AI Pick-and-Place Demonstration.

This script demonstrates pick-and-place manipulation task using the WidowX AI robot.

Usage:
    ~/isaacsim/python.sh scripts/wxai_pick_place.py
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
DEFAULT_CUBE_POSITION = np.array([0.35, -0.15, 0.03])
DEFAULT_CUBE_ORIENTATION = np.array([1, 0, 0, 0])
DEFAULT_TARGET_POSITION = np.array([0.35, 0.15, 0.03])
DEFAULT_HOME_POSITION = np.array([0.2, 0.0, 0.3])
DEFAULT_EVENTS_DT = [80, 50, 10, 50, 80, 50, 10, 50, 80]

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15  # Height clearance above objects in meters
APPROACH_OFFSET = np.array([0.0, 0.0, 0.03])  # Vertical offset for approach in meters
DOWNWARD_ORIENTATION = np.array(
    [[0.7071068, 0.0, 0.7071068, 0.0]]
)  # Downward-facing quaternion [w, x, y, z]

# Scene configuration
ROBOT_USD_PATH = "./assets/robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"
CUBE_SCENE_PATH = "/World/Cube"

# Robot controller configuration
WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]


class WXAIPickPlace:
    """Pick-and-place task with trajectory-based motion control."""

    def __init__(
        self,
        events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        cube_size: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
    ):
        """Initialize pick-and-place task.

        Args:
            events_dt: List of time deltas for events in the task sequence.
            cube_initial_position: Initial position [x, y, z] of the cube in meters.
            cube_initial_orientation: Initial orientation quaternion [w, x, y, z] of the cube.
            cube_size: Size of the cube [width, height, depth] in meters.
            target_position: Target position [x, y, z] for placing the cube in meters.
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
        self.target_position = (
            target_position if target_position is not None else DEFAULT_TARGET_POSITION
        )

        self.events_dt = events_dt if events_dt is not None else DEFAULT_EVENTS_DT

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET
        self.home_position = DEFAULT_HOME_POSITION

        self.cube = None
        self.robot = None
        self.trajectory = None
        self.trajectory_index = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        stage_utils.create_new_stage(template="sunlight")

        # Spawn robot in scene
        stage_utils.add_reference_to_stage(
            usd_path=ROBOT_USD_PATH,
            path=ROBOT_SCENE_PATH,
        )

        self.robot = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.WXAI,
            arm_dof_indices=WXAI_ARM_DOF_INDICES,
            gripper_dof_index=WXAI_GRIPPER_DOF_INDEX,
            default_dof_positions=WXAI_DEFAULT_DOF_POSITIONS,
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
        """Execute one simulation step of the pick-and-place trajectory.

        Returns:
            bool: True if trajectory is in progress, False if complete.
        """
        if self.is_done():
            return False

        if self.trajectory is None:
            self.generate_pick_place_trajectory()

        if self.trajectory_index < len(self.trajectory):
            goal_position, goal_orientation, _ = self.trajectory[self.trajectory_index]

            self.robot.set_end_effector_pose(
                position=goal_position.reshape(1, -1),
                orientation=goal_orientation.reshape(1, -1),
            )

            self.trajectory_index += 1

            phase_boundaries = [0]
            cumulative = 0
            for duration in self.events_dt:
                cumulative += duration
                phase_boundaries.append(cumulative)

            if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                self.robot.close_gripper()
            elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                self.robot.open_gripper()

        return True

    def is_done(self) -> bool:
        """Check if pick-and-place task is complete.

        Returns:
            bool: True if all trajectory waypoints have been executed.
        """
        return self.trajectory is not None and self.trajectory_index >= len(
            self.trajectory
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
        """Reset robot to default pose and clear trajectory."""
        if self.robot is None:
            raise RuntimeError("Cannot reset robot: robot not initialized.")

        self.robot.reset_to_default_pose()
        self.trajectory = None
        self.trajectory_index = 0

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
            n_steps = dt[i]

            # Linear interpolation for each step in this segment
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

    def generate_pick_place_trajectory(self) -> None:
        """Generate complete pick-and-place trajectory from current state.

        Creates a 9-phase trajectory with smooth linear interpolation:
        1. Move to pre-pick position above cube
        2. Descend to pick approach height
        3. Close gripper
        4. Lift cube with clearance
        5. Move to pre-place position above target
        6. Descend to place approach height
        7. Open gripper
        8. Retreat with clearance
        9. Return to home position
        """
        cube_pos = self.cube.get_world_poses()[0].numpy().flatten()
        _, current_ee_pos, _ = self.robot.get_current_state()
        current_ee_pos = current_ee_pos[0]
        key_frames = [
            current_ee_pos,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            cube_pos + self.approach_offset,
            cube_pos + self.approach_offset,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.target_position + np.array([0.0, 0.0, self.clearance_height]),
            self.target_position + self.approach_offset,
            self.target_position + self.approach_offset,
            self.target_position + np.array([0.0, 0.0, self.clearance_height]),
            self.home_position,
        ]

        goal_orientation = DOWNWARD_ORIENTATION[0]
        orientations = [goal_orientation for _ in key_frames]

        self.trajectory = self.make_trajectory(key_frames, orientations, self.events_dt)
        self.trajectory_index = 0


def main():
    print("WidowX AI Pick-and-Place Demo")
    simulation_app.update()

    pick_place = WXAIPickPlace()
    pick_place.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    task_completed = False
    pick_place.reset()

    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            pick_place.forward()

        if pick_place.is_done() and not task_completed:
            print("Task complete")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping pick and place demo...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()

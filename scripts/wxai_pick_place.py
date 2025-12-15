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

This script demonstrates a complete pick-and-place task using the WidowX AI
manipulator with a 9-phase state machine.

Usage:
    ~/isaacsim5.1/isaac-sim.sh scripts/wxai_pick_place.py
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
from wxai_controller import WXAIController  # noqa: E402


class WXAIPickPlace:
    """Pick-and-place task with 9-phase state machine."""

    def __init__(self, events_dt: list[float] | None = None):
        """Initialize task.

        Args:
            events_dt: Duration in steps for each of the 9 phases
        """
        self.cube = None
        self.robot = None

        self.events_dt = events_dt
        if self.events_dt is None:
            self.events_dt = [
                80,  # Pre-pick: above cube
                50,  # Pick approach
                30,  # Grasp
                10,  # Post-pick lift
                50,  # Pre-place: above target
                50,  # Place approach
                30,  # Release
                40,  # Post-place retreat
                80,  # Return home
            ]
        self._event = 0
        self._step = 0

    def setup_scene(
        self,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        cube_size: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> None:
        """Setup scene with robot, cube, and environment."""
        self.cube_initial_position = cube_initial_position
        self.cube_initial_orientation = cube_initial_orientation
        self.target_position = target_position
        self.cube_size = cube_size
        self.offset = offset

        if self.cube_size is None:
            self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        if self.cube_initial_position is None:
            self.cube_initial_position = np.array([0.35, -0.15, 0.05])
        if self.cube_initial_orientation is None:
            self.cube_initial_orientation = np.array([1, 0, 0, 0])
        if self.target_position is None:
            self.target_position = np.array([0.35, 0.25, 0.05])
        if self.offset is None:
            self.offset = np.array([0.0, 0.0, 0.0])
        self.target_position = self.target_position + self.offset

        stage_utils.create_new_stage(template="sunlight")

        self.robot = WXAIController(robot_path="/World/wxai_robot", create_robot=True)
        self.end_effector_link = self.robot.end_effector_link

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

        cube_shape = Cube(
            paths="/World/Cube",
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
        """Execute one step of the pick-and-place state machine."""
        if self.is_done():
            return False

        goal_orientation = self.robot.get_downward_orientation()

        # Phase 0: Pre-pick (above cube)
        if self._event == 0:
            if self._step == 0:
                print("Phase 0: Moving to pre-pick position...")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            z_offset = 0.10
            goal_position = np.array(
                [cube_pos[0, 0], cube_pos[0, 1], cube_pos[0, 2] + z_offset]
            )
            self.robot.set_end_effector_pose(
                position=goal_position, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[0]:
                self._event += 1
                self._step = 0

        # Phase 1: Pick approach
        elif self._event == 1:
            if self._step == 0:
                print("Phase 1: Approaching pick position...")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.04])
            self.robot.set_end_effector_pose(
                position=goal_position, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[1]:
                self._event += 1
                self._step = 0

        # Phase 2: Grasp
        elif self._event == 2:
            if self._step == 0:
                print("Phase 2: Grasping cube...")

            self.robot.close_gripper()

            self._step += 1
            if self._step >= self.events_dt[2]:
                self._event += 1
                self._step = 0

        # Phase 3: Post-pick lift
        elif self._event == 3:
            if self._step == 0:
                print("Phase 3: Lifting cube...")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            z_offset = 0.25
            goal_position = cube_pos + np.array([0.0, 0.0, z_offset])
            self.robot.set_end_effector_pose(
                position=goal_position, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[3]:
                self._event += 1
                self._step = 0

        # Phase 4: Pre-place (above target)
        elif self._event == 4:
            if self._step == 0:
                print("Phase 4: Moving to pre-place position...")

            z_offset = 0.25
            goal_position = np.array(
                [
                    self.target_position[0],
                    self.target_position[1],
                    self.target_position[2] + z_offset,
                ]
            )
            self.robot.set_end_effector_pose(
                position=goal_position, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[4]:
                self._event += 1
                self._step = 0

        # Phase 5: Place approach
        elif self._event == 5:
            if self._step == 0:
                print("Phase 5: Approaching place position...")

            target_pos = self.target_position + np.array([0.0, 0.0, 0.03])
            self.robot.set_end_effector_pose(
                position=target_pos, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[5]:
                self._event += 1
                self._step = 0

        # Phase 6: Release
        elif self._event == 6:
            if self._step == 0:
                print("Phase 6: Releasing cube...")

            self.robot.open_gripper()

            self._step += 1
            if self._step >= self.events_dt[6]:
                self._event += 1
                self._step = 0

        # Phase 7: Post-place retreat
        elif self._event == 7:
            if self._step == 0:
                print("Phase 7: Retreating...")

            z_offset = 0.15
            goal_position = self.target_position + np.array([0.0, 0.0, z_offset])
            self.robot.set_end_effector_pose(
                position=goal_position, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[7]:
                self._event += 1
                self._step = 0

        # Phase 8: Return home
        elif self._event == 8:
            if self._step == 0:
                print("Phase 8: Returning home...")

            home_position = np.array([0.2, 0.0, 0.3])
            self.robot.set_end_effector_pose(
                position=home_position, orientation=goal_orientation
            )

            self._step += 1
            if self._step >= self.events_dt[8]:
                self._event += 1
                self._step = 0

        return True

    def is_done(self) -> bool:
        """Returns True if all phases complete."""
        return self._event >= len(self.events_dt)

    def reset(
        self,
        cube_position: np.ndarray | None = None,
        cube_orientation: np.ndarray | None = None,
    ):
        """Reset task to initial state."""
        print("Resetting pick-and-place...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        print("Reset complete")

    def reset_robot(self):
        """Reset robot and state machine."""
        if self.robot is not None:
            self.robot.reset_to_default_pose()
            self._event = 0
            self._step = 0

    def reset_cube(
        self, position: np.ndarray | None = None, orientation: np.ndarray | None = None
    ):
        """Reset cube to initial pose."""
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
    """Run pick-and-place demonstration."""
    print("WidowX AI Pick-and-Place Demo")
    simulation_app.update()

    pick_place = WXAIPickPlace()
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
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()

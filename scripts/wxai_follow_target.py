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
WidowX AI Follow Target Demonstration.

This script demonstrates continuous target tracking where the robot's end
effector follows a manually movable target cube.

Usage:
    ~/isaacsim/python.sh scripts/wxai_follow_target.py
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
from isaacsim.core.experimental.prims import GeomPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController  # noqa: E402

# Default target configuration
DEFAULT_TARGET_POSITION = np.array([0.3, 0.0, 0.2])
DEFAULT_TARGET_ORIENTATION = np.array([1, 0, 0, 0])
DEFAULT_TARGET_SIZE = np.array([0.0515, 0.0515, 0.0515])

# Scene configuration
ROBOT_USD_PATH = "./assets/robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"
TARGET_SCENE_PATH = "/World/TargetCube"

# Robot controller configuration
WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]


class WXAIFollowTarget:
    """Real-time target tracking demonstration.

    Continuously tracks a movable target cube's pose, commanding the robot's
    end effector to match the target's position and orientation.
    """

    def __init__(
        self,
        target_initial_position: np.ndarray | None = None,
        target_initial_orientation: np.ndarray | None = None,
        target_size: np.ndarray | None = None,
    ):
        """Initialize target tracking task."""
        self.target_initial_position = (
            target_initial_position
            if target_initial_position is not None
            else DEFAULT_TARGET_POSITION
        )
        self.target_initial_orientation = (
            target_initial_orientation
            if target_initial_orientation is not None
            else DEFAULT_TARGET_ORIENTATION
        )
        self.target_size = (
            target_size if target_size is not None else DEFAULT_TARGET_SIZE
        )

        self.target = None
        self.robot = None

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, target cube, and environment."""
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

        target_shape = Cube(
            paths=TARGET_SCENE_PATH,
            positions=self.target_initial_position,
            orientations=self.target_initial_orientation,
            sizes=[1.0],
            scales=self.target_size,
            reset_xform_op_properties=True,
        )

        self.target = GeomPrim(paths=target_shape.paths)
        target_shape.apply_visual_materials(visual_material)

    def forward(self) -> None:
        """Execute one tracking step by commanding end effector to current target pose."""
        target_position, target_orientation = self.target.get_world_poses()
        target_position = target_position.numpy().flatten()
        target_orientation = target_orientation.numpy().flatten()

        self.robot.set_end_effector_pose(
            position=target_position, orientation=target_orientation
        )

    def reset(
        self,
        target_position: np.ndarray | None = None,
        target_orientation: np.ndarray | None = None,
    ) -> None:
        """Reset robot and target to initial state."""
        if self.robot is not None:
            self.robot.reset_to_default_pose()

        if self.target is not None:
            reset_position = (
                target_position
                if target_position is not None
                else self.target_initial_position
            )
            reset_orientation = (
                target_orientation
                if target_orientation is not None
                else self.target_initial_orientation
            )
            self.target.set_world_poses(
                positions=reset_position.reshape(1, -1),
                orientations=reset_orientation.reshape(1, -1),
            )


def main():
    simulation_app.update()

    follow_target = WXAIFollowTarget()
    follow_target.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    reset_needed = True

    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            if reset_needed:
                follow_target.reset()
                reset_needed = False

            follow_target.forward()

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping follow target demo...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()

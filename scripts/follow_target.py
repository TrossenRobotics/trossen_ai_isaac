# Copyright 2026 Trossen Robotics
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
Follow Target Demonstration.

This script demonstrates continuous target tracking where the end effector of the
WidowX AI or WidowX Pro robot follows a manually movable target cube.

Usage:
    ~/isaacsim/python.sh scripts/follow_target.py --robot wxai
    ~/isaacsim/python.sh scripts/follow_target.py --robot wxpro
"""

from __future__ import annotations

import argparse
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
DEFAULT_TARGET_POSITION = np.array([0.35, 0.0, 0.3])
DEFAULT_TARGET_ORIENTATION = np.array([1, 0, 0, 0])
DEFAULT_TARGET_SIZE = np.array([0.0515, 0.0515, 0.0515])

# Scene configuration
ROBOT_SCENE_PATH = "/World/robot"
GROUND_SCENE_PATH = "/World/ground"
TARGET_SCENE_PATH = "/World/TargetCube"

# Supported robots
ROBOTS = {
    "wxai": {
        "usd_path": "./assets/robots/wxai/wxai_base.usd",
        "gripper_dof_index": 6,
        "ee_offset": np.array([0.156062, 0.0, 0.0]),
    },
    "wxpro": {
        "usd_path": "./assets/robots/wxpro/wxpro.usd",
        # left_carriage_joint (DOF order: right_carriage_joint, left_carriage_joint); the right finger mimics it
        "gripper_dof_index": 7,
        "ee_offset": np.array([0.218, 0.0, 0.0]),
    },
}
DEFAULT_ROBOT = "wxpro"

# Robot controller configuration
ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]

# IK configuration
IK_SCALE = 1.0
IK_DAMPING = 0.03


class FollowTarget:
    """Real-time target tracking demonstration.

    Continuously tracks a movable target cube's pose, commanding the robot's
    end effector to match the target's position and orientation.
    """

    def __init__(
        self,
        robot: str = DEFAULT_ROBOT,
        target_initial_position: np.ndarray | None = None,
        target_initial_orientation: np.ndarray | None = None,
        target_size: np.ndarray | None = None,
    ):
        """Initialize target tracking task.

        Args:
            robot: Robot name, one of ROBOTS.
            target_initial_position: Initial position [x, y, z] of the target in meters.
            target_initial_orientation: Initial orientation quaternion [w, x, y, z] of the target.
            target_size: Target cube size [x, y, z] in meters.
        """
        if robot not in ROBOTS:
            raise ValueError(f"Unknown robot '{robot}', expected one of {list(ROBOTS)}")
        self.robot_name = robot
        self.robot_config = ROBOTS[robot]

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
            usd_path=self.robot_config["usd_path"],
            path=ROBOT_SCENE_PATH,
        )

        self.robot = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType(self.robot_name),
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
            arm_dof_indices=ARM_DOF_INDICES,
            gripper_dof_index=self.robot_config["gripper_dof_index"],
            default_dof_positions=DEFAULT_DOF_POSITIONS,
            ee_offset=self.robot_config["ee_offset"],
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
        if self.robot is None:
            raise RuntimeError("Cannot reset robot: robot not initialized.")
        if self.target is None:
            raise RuntimeError("Cannot reset target: target not initialized.")

        self.robot.reset_to_default_pose()

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
    parser = argparse.ArgumentParser(
        description="Follow a movable target cube with the end effector."
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=list(ROBOTS),
        default=DEFAULT_ROBOT,
        help="Robot to run.",
    )
    args = parser.parse_args()

    simulation_app.update()

    follow_target = FollowTarget(robot=args.robot)
    follow_target.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    follow_target.reset()

    while simulation_app.is_running():
        if SimulationManager.is_simulating():
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

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
WidowX AI Robot Controller.

This module provides a controller for the WidowX AI manipulator with
differential inverse kinematics and gripper control.
"""

from typing import Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.experimental.utils.impl.transform import (
    quaternion_conjugate,
    quaternion_multiplication,
)
from scipy.spatial.transform import Rotation


class WXAIController(Articulation):
    """Controller for WidowX AI manipulator with differential IK and gripper control."""

    def __init__(
        self,
        robot_path: str = "/World/wxai_robot",
        usd_path: str = None,
        create_robot: bool = True,
        end_effector_link: Optional[RigidPrim] = None,
    ):
        """Initialize controller.

        Args:
            robot_path: Scene path for robot
            usd_path: Path to robot USD file (default: ./assets/robots/wxai/wxai_base.usd)
            create_robot: Whether to create robot or use existing
            end_effector_link: Custom end effector link (default: link_6)
        """
        if create_robot:
            if usd_path is None:
                usd_path = "./assets/robots/wxai/wxai_base.usd"

            stage_utils.add_reference_to_stage(
                usd_path=usd_path,
                path=robot_path,
            )

        super().__init__(robot_path)

        if end_effector_link is None:
            self.end_effector_link = RigidPrim(f"{robot_path}/link_6")
        else:
            self.end_effector_link = end_effector_link

        self.ee_offset = self.get_gripper_center_offset()

        if create_robot:
            self.set_default_state(
                dof_positions=[0.0, 1.2, 1.0, 0.0, 0.0, 0.0, 0.044, 0.044]
            )

        self.end_effector_link_index = self.get_link_indices("link_6").list()[0]
        self.gripper_open_position = 0.044
        self.gripper_closed_position = 0.015

    def differential_inverse_kinematics(
        self,
        jacobian_end_effector: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute joint delta using damped least-squares IK."""
        scale = 1.0
        damping = 0.05

        goal_orientation = (
            current_orientation if goal_orientation is None else goal_orientation
        )

        goal_quat_wp = wp.from_numpy(goal_orientation, dtype=wp.float32)
        current_quat_wp = wp.from_numpy(current_orientation, dtype=wp.float32)
        current_quat_conjugate_wp = quaternion_conjugate(current_quat_wp)
        q_wp = quaternion_multiplication(goal_quat_wp, current_quat_conjugate_wp)
        q_np = q_wp.numpy()

        error = np.expand_dims(
            np.concatenate(
                [goal_position - current_position, q_np[:, 1:] * np.sign(q_np[:, [0]])],
                axis=-1,
            ),
            axis=2,
        )

        transpose = np.swapaxes(jacobian_end_effector, 1, 2)
        lmbda = np.eye(jacobian_end_effector.shape[1]) * (damping**2)
        return (
            scale
            * transpose
            @ np.linalg.inv(jacobian_end_effector @ transpose + lmbda)
            @ error
        ).squeeze(-1)

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (joint_positions, ee_position, ee_orientation)."""
        current_dof_positions = self.get_dof_positions().numpy()
        link6_position, link6_orientation = self.end_effector_link.get_world_poses()
        link6_position = link6_position.numpy()
        link6_orientation = link6_orientation.numpy()

        current_end_effector_position = self._transform_offset_to_world(
            link6_position, link6_orientation, self.ee_offset
        )
        current_end_effector_orientation = link6_orientation

        return (
            current_dof_positions,
            current_end_effector_position,
            current_end_effector_orientation,
        )

    def _transform_offset_to_world(
        self, position: np.ndarray, orientation: np.ndarray, offset: np.ndarray
    ) -> np.ndarray:
        """Transform local offset to world coordinates."""
        result_positions = []
        for i in range(position.shape[0]):
            quat = orientation[i]
            quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
            rotation = Rotation.from_quat(quat_scipy)
            offset_world = rotation.apply(offset)
            result_positions.append(position[i] + offset_world)

        return np.array(result_positions)

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        """Command end effector to target pose using differential IK.

        Args:
            position: Target position [x, y, z]
            orientation: Target quaternion [w, x, y, z]
        """

        (
            current_dof_positions,
            current_end_effector_position,
            current_end_effector_orientation,
        ) = self.get_current_state()

        if position.ndim == 1:
            position = position.reshape(1, -1)

        goal_link6_position = self._transform_ee_goal_to_link6(position, orientation)

        jacobian_matrices = self.get_jacobian_matrices().numpy()
        jacobian_end_effector = jacobian_matrices[
            :, self.end_effector_link_index - 1, :, :6
        ]

        link6_position, link6_orientation = self.end_effector_link.get_world_poses()
        link6_position = link6_position.numpy()
        link6_orientation = link6_orientation.numpy()

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_end_effector,
            current_position=link6_position,
            current_orientation=link6_orientation,
            goal_position=goal_link6_position,
            goal_orientation=orientation,
        )

        dof_position_targets = current_dof_positions[:, :6] + delta_dof_positions
        self.set_dof_position_targets(dof_position_targets, dof_indices=list(range(6)))

    def _transform_ee_goal_to_link6(
        self, ee_position: np.ndarray, ee_orientation: np.ndarray
    ) -> np.ndarray:
        """Convert end effector goal to link_6 frame."""
        result_positions = []
        for i in range(ee_position.shape[0]):
            quat = ee_orientation[i]
            quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
            rotation = Rotation.from_quat(quat_scipy)
            offset_world = rotation.apply(self.ee_offset)
            result_positions.append(ee_position[i] - offset_world)

        return np.array(result_positions)

    def open_gripper(self) -> None:
        """Open gripper."""
        position = np.array([[self.gripper_open_position]])
        self.set_dof_position_targets(position, dof_indices=[6])

    def close_gripper(self) -> None:
        """Close gripper."""
        position = np.array([[self.gripper_closed_position]])
        self.set_dof_position_targets(position, dof_indices=[6])

    def set_gripper_position(self, position: float) -> None:
        """Set gripper opening (0.015 = closed, 0.044 = open)."""
        position_array = np.array([[position]])
        self.set_dof_position_targets(position_array, dof_indices=[6])

    def get_downward_orientation(self) -> np.ndarray:
        """Returns downward-facing orientation quaternion."""
        return np.array([[0.7071068, 0.0, 0.7071068, 0.0]])

    def get_gripper_center_offset(self) -> np.ndarray:
        """Returns gripper center offset from link_6 in meters."""
        return np.array([0.1055, 0.0, 0.0])

    def reset_to_default_pose(self) -> None:
        """Reset to default pose."""
        default_positions = np.array([[0.0, 1.2, 1.0, 0.0, 0.0, 0.0, 0.044, 0.044]])
        self.set_dof_positions(default_positions)
        self.set_dof_position_targets(default_positions)

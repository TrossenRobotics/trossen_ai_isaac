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
Trossen AI Robot Controller.

This module provides a controller for Trossen AI robots with
differential inverse kinematics and gripper control.
Supports:
- WidowX AI
- Stationary AI
"""

import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.experimental.utils.impl.transform import (
    quaternion_conjugate,
    quaternion_multiplication,
)
from scipy.spatial.transform import Rotation

# Robot configuration constants
DEFAULT_ROBOT_PATH = "/World/wxai_robot"
END_EFFECTOR_LINK_NAME = "link_6"

# End effector offset from link_6 in meters [x, y, z]
EE_OFFSET = np.array([0.1055, 0.0, 0.0])

# Inverse kinematics parameters
DEFAULT_IK_SCALE = 0.5  # Scaling factor for joint velocity commands
DEFAULT_IK_DAMPING = 0.03  # Damping for singularity robustness

# Gripper position limits in meters
GRIPPER_OPEN_POSITION = 0.044
GRIPPER_CLOSED_POSITION = 0.022  # Gripper closed around the cube

# Default joint configuration for WidowX AI (8 DOFs: 6 arm joints + 2 gripper fingers)
DEFAULT_DOF_POSITIONS_WXAI = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]

# Default joint configuration for Stationary AI (16 DOFs: 2 arms interleaved + 4 gripper fingers)
DEFAULT_DOF_POSITIONS_STATIONARY_AI = [
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


class TrossenAIController(Articulation):
    """Controller for Trossen AI robots with differential IK and gripper control.

    Supported robots:
    - WidowX AI: 8 DOFs (6 arm joints + 2 gripper fingers)
    - Stationary AI: 16 DOFs (6 + 6 left and right arm joints, 2 + 2 left and right gripper fingers)
    """

    def __init__(
        self,
        robot_path: str = DEFAULT_ROBOT_PATH,
        robot_type: str = "wxai",
        end_effector_link: RigidPrim | None = None,
        ik_scale: float = DEFAULT_IK_SCALE,
        ik_damping: float = DEFAULT_IK_DAMPING,
        arm_dof_indices: list[int] | None = None,
        gripper_dof_index: int | None = None,
        default_dof_positions: list[float] | None = None,
    ):
        """Initialize Trossen AI robot controller.

        Args:
            robot_path: USD scene path for the robot that already exists in the scene.
            robot_type: Type of robot (e.g., "wxai", "stationary_ai"). Default: "wxai"
            end_effector_link: End effector link. If None, defaults to 'link_6'.
            ik_scale: Scaling factor for IK joint velocity commands (0.0-1.0). Default: 0.5
            ik_damping: Damping factor for singularity robustness (0.0-0.1). Default: 0.03
            arm_dof_indices: DOF indices for the arm joints. Required.
            gripper_dof_index: DOF index for the gripper. Required.
            default_dof_positions: Default joint configuration for reset. Required.
        """
        super().__init__(robot_path)

        self.robot_type = robot_type

        if arm_dof_indices is not None:
            self.arm_dof_indices = arm_dof_indices
        else:
            raise ValueError("arm_dof_indices must be provided")

        if gripper_dof_index is not None:
            self.gripper_dof_index = gripper_dof_index
        else:
            raise ValueError("gripper_dof_index must be provided")

        if default_dof_positions is not None:
            self.default_dof_positions = default_dof_positions
        else:
            raise ValueError("default_dof_positions must be provided")

        if end_effector_link is not None:
            self.end_effector_link = end_effector_link
            link_name = end_effector_link.paths[0].split("/")[-1]
        else:
            self.end_effector_link = RigidPrim(f"{robot_path}/{END_EFFECTOR_LINK_NAME}")
            link_name = END_EFFECTOR_LINK_NAME

        self.end_effector_link_index = self.get_link_indices(link_name).list()[0]
        self.ee_offset = EE_OFFSET

        self.ik_scale = ik_scale
        self.ik_damping = ik_damping

        self.set_default_state(dof_positions=self.default_dof_positions)

    def differential_inverse_kinematics(
        self,
        jacobian_end_effector: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint velocity commands using damped least-squares IK.

        Args:
            jacobian_end_effector: End effector Jacobian matrix (batch_size, 6, num_joints).
            current_position: Current end effector position (batch_size, 3).
            current_orientation: Current orientation quaternion (batch_size, 4) [w, x, y, z].
            goal_position: Target position (batch_size, 3).
            goal_orientation: Target orientation quaternion (batch_size, 4). Uses current if None.

        Returns:
            Joint position deltas (batch_size, num_joints).
        """
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
        damping_matrix = np.eye(jacobian_end_effector.shape[1]) * (self.ik_damping**2)
        return (
            self.ik_scale
            * transpose
            @ np.linalg.inv(jacobian_end_effector @ transpose + damping_matrix)
            @ error
        ).squeeze(-1)

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current robot state.

        Returns:
            Tuple of (joint_positions, end_effector_position, end_effector_orientation).
            Shapes: (batch_size, num_dofs), (batch_size, 3), (batch_size, 4)
        """
        current_dof_positions = self.get_dof_positions().numpy()
        wrist_position, wrist_orientation = self.end_effector_link.get_world_poses()
        wrist_position = wrist_position.numpy()
        wrist_orientation = wrist_orientation.numpy()

        current_end_effector_position = self._transform_offset_to_world(
            wrist_position, wrist_orientation, self.ee_offset
        )
        current_end_effector_orientation = wrist_orientation

        return (
            current_dof_positions,
            current_end_effector_position,
            current_end_effector_orientation,
        )

    def _transform_offset_to_world(
        self, position: np.ndarray, orientation: np.ndarray, offset: np.ndarray
    ) -> np.ndarray:
        """Transform local offset to world coordinates.

        Args:
            position: Position in world frame (batch_size, 3).
            orientation: Orientation quaternion [w, x, y, z] (batch_size, 4).
            offset: Local offset vector [x, y, z].

        Returns:
            World position with offset applied (batch_size, 3).
        """
        quat_scipy = orientation[:, [1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat_scipy)
        offset_world = rotation.apply(np.tile(offset, (position.shape[0], 1)))
        return position + offset_world

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        """Command end effector to target Cartesian pose using differential IK.

        Args:
            position: Target position [x, y, z] in meters. Shape: (3,) or (1, 3).
            orientation: Target orientation quaternion [w, x, y, z]. Shape: (4,) or (1, 4).
        """

        (
            current_dof_positions,
            _,
            _,
        ) = self.get_current_state()

        position = position.reshape(1, -1)
        orientation = orientation.reshape(1, -1)

        goal_wrist_position = self._transform_ee_to_wrist_frame(position, orientation)

        jacobian_matrices = self.get_jacobian_matrices().numpy()
        jacobian_full = jacobian_matrices[:, self.end_effector_link_index - 1, :, :]

        if self.robot_type == "wxai":
            jacobian_end_effector = jacobian_full[:, :, :6]
        else:
            jacobian_end_effector = jacobian_full[:, :, self.arm_dof_indices]

        wrist_position, wrist_orientation = self.end_effector_link.get_world_poses()
        wrist_position = wrist_position.numpy()
        wrist_orientation = wrist_orientation.numpy()

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_end_effector,
            current_position=wrist_position,
            current_orientation=wrist_orientation,
            goal_position=goal_wrist_position,
            goal_orientation=orientation,
        )

        if self.robot_type == "wxai":
            dof_position_targets = current_dof_positions[:, :6] + delta_dof_positions
            self.set_dof_position_targets(
                dof_position_targets, dof_indices=list(range(6))
            )
        else:
            dof_position_targets = (
                current_dof_positions[:, self.arm_dof_indices] + delta_dof_positions
            )
            self.set_dof_position_targets(
                dof_position_targets, dof_indices=self.arm_dof_indices
            )

    def _transform_ee_to_wrist_frame(
        self, ee_position: np.ndarray, ee_orientation: np.ndarray
    ) -> np.ndarray:
        """Convert end effector goal to wrist (link_6) frame.

        Args:
            ee_position: End effector position (batch_size, 3).
            ee_orientation: End effector orientation quaternion [w, x, y, z] (batch_size, 4).

        Returns:
            Wrist position in world frame (batch_size, 3).
        """
        quat_scipy = ee_orientation[:, [1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat_scipy)
        offset_world = rotation.apply(
            np.tile(self.ee_offset, (ee_position.shape[0], 1))
        )
        return ee_position - offset_world

    def open_gripper(self) -> None:
        """Open the gripper to maximum width (0.044 meters)."""
        self.set_gripper_position(GRIPPER_OPEN_POSITION)

    def close_gripper(self) -> None:
        """Close the gripper around object (0.022 meters)."""
        self.set_gripper_position(GRIPPER_CLOSED_POSITION)

    def set_gripper_position(self, position: float) -> None:
        """Set gripper opening width.

        Args:
            position: Gripper width in meters (0.022 = closed, 0.044 = open).
        """
        self.set_dof_position_targets(
            np.array([[position]]), dof_indices=[self.gripper_dof_index]
        )

    def reset_to_default_pose(self) -> None:
        """Reset robot to default configuration with all joints at zero and gripper open."""
        default_positions = np.array([self.default_dof_positions])
        self.set_dof_positions(default_positions)
        self.set_dof_position_targets(default_positions)

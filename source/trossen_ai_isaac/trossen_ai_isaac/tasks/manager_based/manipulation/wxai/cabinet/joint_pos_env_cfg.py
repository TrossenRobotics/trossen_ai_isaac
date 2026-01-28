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

"""Joint position control environment configuration for WXAI cabinet (drawer) task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
)

from trossen_ai_isaac.tasks.manager_based.manipulation.assets import WXAI_BASE_CFG


@configclass
class WXAICabinetEnvCfg(CabinetEnvCfg):
    """Configuration for WXAI cabinet environment with joint position control."""

    def __post_init__(self):
        super().__post_init__()

        # Set WXAI as robot - raised on platform
        self.scene.robot = WXAI_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (
            0.0,
            0.0,
            0.3,
        )  # Raise robot to sit on platform

        # Add blue platform under robot
        self.scene.platform = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Platform",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),  # Width, depth, height
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.4, 0.9)
                ),  # Blue color
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.15)
            ),  # Platform center at half height
        )

        # Set Actions for the specific robot type (WXAI)
        # Arm action: 6 DOF arm joints
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[0-5]"],
            scale=1.0,
            use_default_offset=True,
        )

        # Gripper action: parallel gripper with left_carriage_joint
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_carriage_joint"],
            open_command_expr={"left_carriage_joint": 0.044},  # Open position (meters)
            close_command_expr={"left_carriage_joint": 0.0},  # Closed position
        )

        # Configure end-effector frame transformer
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
                prim_path="/Visuals/EndEffectorFrameTransformer"
            ),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_gripper_link",
                    name="ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/carriage_left",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/carriage_right",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # Override rewards for WXAI gripper parameters
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = [
            "left_carriage_joint"
        ]


@configclass
class WXAICabinetEnvCfg_PLAY(WXAICabinetEnvCfg):
    """Configuration for WXAI cabinet environment in play mode."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

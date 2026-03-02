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

"""Joint position control environment configuration for WXAI reach task."""

import math

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from trossen_ai_isaac.tasks.manager_based.manipulation.assets import WXAI_HIGH_PD_CFG

##
# Environment configuration
##


@configclass
class WXAIReachEnvCfg(ReachEnvCfg):
    """Configuration for WXAI reach environment with joint position control."""

    def __post_init__(self):
        super().__post_init__()

        # Set robot configuration
        self.scene.robot = WXAI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Configure end-effector tracking rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "ee_gripper_link"
        ]
        self.rewards.end_effector_position_tracking_fine_grained.params[
            "asset_cfg"
        ].body_names = ["ee_gripper_link"]
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = ["ee_gripper_link"]

        # Prioritize position over orientation
        self.rewards.end_effector_position_tracking.weight = -0.4
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.2
        self.rewards.end_effector_orientation_tracking.weight = -0.17

        # Configure arm action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[0-5]"],
            scale=0.5,
            use_default_offset=True,
        )

        # Configure end-effector pose command ranges
        self.commands.ee_pose.body_name = "ee_gripper_link"
        self.commands.ee_pose.ranges.roll = (-math.pi / 4, math.pi / 4)
        self.commands.ee_pose.ranges.pitch = (0, math.pi / 4)
        self.commands.ee_pose.ranges.yaw = (-math.pi / 4, math.pi / 4)
        self.commands.ee_pose.ranges.pos_x = (0.15, 0.35)
        self.commands.ee_pose.ranges.pos_z = (0.02, 0.30)


@configclass
class WXAIReachEnvCfg_PLAY(WXAIReachEnvCfg):
    """Configuration for WXAI reach environment in play mode."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

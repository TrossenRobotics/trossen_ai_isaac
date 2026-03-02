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

"""Shared leader arm interface for teleoperation scripts."""

from __future__ import annotations

import logging
import time

import numpy as np
import trossen_arm

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6
LEADER_HOME_POSITION = np.array([0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0])


class LeaderArmHardware:
    """Interface to a real Trossen WXAI leader arm.

    Connects to the arm over Ethernet, moves it to a home position,
    then switches to gravity compensation so the operator can move
    it freely.  Joint positions and gripper opening are read each
    step via :meth:`get_state`.
    """

    def __init__(self, ip: str = "192.168.1.2"):
        self.ip = ip
        self.driver = None

    def connect(self) -> None:
        """Connect to the leader arm, home it, and enable gravity compensation."""
        logger.info("Connecting to leader arm at %s...", self.ip)
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            self.ip,
            False,
        )

        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(LEADER_HOME_POSITION, 2.0, True)

        time.sleep(0.5)
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver.set_all_external_efforts(
            [0.0] * self.driver.get_num_joints(), 0.0, False
        )

    def get_state(self) -> tuple[np.ndarray, float]:
        """Return (arm_positions, gripper_position).

        arm_positions: 6 joint angles in radians.
        gripper_position: finger opening in meters (0.0–0.044).
        """
        positions = self.driver.get_all_positions()
        return np.array(positions[:NUM_ARM_JOINTS]), float(positions[NUM_ARM_JOINTS])

    def cleanup(self) -> None:
        """Return to home, then park at zero and disconnect."""
        if self.driver is None:
            return
        try:
            self.driver.set_all_modes(trossen_arm.Mode.position)
            self.driver.set_all_positions(LEADER_HOME_POSITION, 2.0, True)
            self.driver.set_all_positions(
                np.zeros(self.driver.get_num_joints()), 2.0, True
            )
            self.driver.cleanup()
        except Exception as e:
            logger.warning("Error during leader arm cleanup: %s", e)

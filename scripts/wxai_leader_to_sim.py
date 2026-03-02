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

"""Control the simulated WXAI robot in Isaac Sim using a real leader arm.

Reads joint positions from the leader arm and maps them 1:1 to the sim robot.
The leader runs in gravity compensation mode for free manual movement.

Usage:
    ~/isaacsim/python.sh scripts/wxai_leader_to_sim.py
    ~/isaacsim/python.sh scripts/wxai_leader_to_sim.py --leader_ip 192.168.1.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

parser = argparse.ArgumentParser(
    description="Control simulated WXAI arm using a real leader arm."
)
parser.add_argument(
    "--leader_ip",
    type=str,
    default="192.168.1.2",
    help="IP address of the leader arm. Default: 192.168.1.2",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run in headless mode (no GUI).",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController  # noqa: E402
from leader_arm import NUM_ARM_JOINTS, LeaderArmHardware  # noqa: E402

logger = logging.getLogger(__name__)

ROBOT_USD_PATH = "./assets/robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"

WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]


def setup_scene() -> TrossenAIController:
    stage_utils.create_new_stage(template="sunlight")

    stage_utils.add_reference_to_stage(
        usd_path=ROBOT_USD_PATH,
        path=ROBOT_SCENE_PATH,
    )

    robot = TrossenAIController(
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

    return robot


def main():
    leader = LeaderArmHardware(ip=args.leader_ip)
    leader.connect()

    robot = setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    robot.reset_to_default_pose()
    simulation_app.update()

    print("Teleoperation active. Ctrl+C to stop.")

    step = 0
    try:
        while simulation_app.is_running():
            if SimulationManager.is_simulating():
                arm_positions, gripper_position = leader.get_state()
                gripper_position = max(0.0, min(0.044, gripper_position))

                # 8 DOFs: 6 arm + left_carriage + right_carriage (mimic)
                dof_targets = np.zeros((1, 8))
                dof_targets[0, :NUM_ARM_JOINTS] = arm_positions
                dof_targets[0, NUM_ARM_JOINTS] = gripper_position
                dof_targets[0, NUM_ARM_JOINTS + 1] = gripper_position

                robot.set_dof_position_targets(dof_targets)

                step += 1
                if step % 30 == 0:
                    joints_str = "  ".join(
                        f"J{i}:{np.degrees(p):+7.1f}\u00b0"
                        for i, p in enumerate(arm_positions)
                    )
                    print(
                        f"\r[{step:6d}] {joints_str}  Grip:{gripper_position * 1000:.1f}mm    ",
                        end="",
                        flush=True,
                    )

            simulation_app.update()

    except KeyboardInterrupt:
        pass
    finally:
        leader.cleanup()
        simulation_app.close()


if __name__ == "__main__":
    main()

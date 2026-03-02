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

"""Teleoperate Isaac Lab WXAI environments with a Trossen leader arm.

Reads joint positions from the leader arm and maps them to actions for
reach, lift, and cabinet tasks using joint-position or IK control modes.
The action type (joint_pos / ik_abs / ik_rel) is auto-detected from the
task name.
"""

from __future__ import annotations

import argparse
import logging

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Teleoperate Isaac Lab WXAI environments with a Trossen leader arm."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-WXAI-v0",
    help="Isaac Lab task name. Default: Isaac-Lift-Cube-WXAI-v0",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate. Default: 1",
)
parser.add_argument(
    "--leader_ip",
    type=str,
    default="192.168.1.2",
    help="IP address of the leader arm. Default: 192.168.1.2",
)
parser.add_argument(
    "--gripper_threshold",
    type=float,
    default=0.022,
    help="Gripper position threshold for open/close (meters). Default: 0.022",
)

# Add AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the Omniverse application
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch

# Register Trossen AI tasks
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from leader_arm import NUM_ARM_JOINTS, LeaderArmHardware  # noqa: E402

logger = logging.getLogger(__name__)

# Action scales matching the environment action configs
JOINT_POS_ACTION_SCALE = 0.5  # JointPositionActionCfg
IK_REL_ACTION_SCALE = 0.01  # DifferentialInverseKinematicsActionCfg (relative)
IK_ABS_ACTION_SCALE = 1.0  # DifferentialInverseKinematicsActionCfg (absolute)

# Tracks previous cartesian pose for IK relative delta computation
_last_cartesian: dict = {}


def detect_action_type(task_name: str) -> str:
    """Return 'joint_pos', 'ik_abs', or 'ik_rel' based on the task name."""
    task_lower = task_name.lower()
    if "ik-abs" in task_lower:
        return "ik_abs"
    elif "ik-rel" in task_lower:
        return "ik_rel"
    else:
        return "joint_pos"


def has_gripper_action(task_name: str) -> bool:
    """True if the task includes a gripper action dimension."""
    return any(keyword in task_name for keyword in ("Lift", "Cabinet"))


def compute_joint_pos_action(
    arm_positions: np.ndarray,
    gripper_position: float,
    include_gripper: bool,
    gripper_threshold: float,
) -> np.ndarray:
    """Convert leader arm joint positions to a joint-position environment action.

    The environment uses ``JointPositionActionCfg(scale=0.5, use_default_offset=True)``
    which applies ``target = default + scale * action``.  Since the WXAI defaults are
    all zeros this simplifies to ``action = desired / scale``.
    """
    arm_action = arm_positions / JOINT_POS_ACTION_SCALE

    if include_gripper:
        gripper_action = 1.0 if gripper_position > gripper_threshold else -1.0
        return np.concatenate([arm_action, [gripper_action]])
    return arm_action


def compute_ik_abs_action(
    gripper_position: float,
    include_gripper: bool,
    gripper_threshold: float,
    leader_interface,
) -> np.ndarray:
    """Convert leader arm FK pose to an IK-absolute environment action.

    Queries the driver's forward kinematics for the end-effector cartesian
    pose ``[x, y, z, rx, ry, rz]`` and divides by the action scale so the
    environment's IK controller receives the desired absolute pose.
    """
    robot_output = leader_interface.driver.get_robot_output()
    cartesian = np.array(robot_output.cartesian.positions)
    # cartesian = [x, y, z, rx, ry, rz]
    arm_action = cartesian / IK_ABS_ACTION_SCALE

    if include_gripper:
        gripper_action = 1.0 if gripper_position > gripper_threshold else -1.0
        return np.concatenate([arm_action, [gripper_action]])
    return arm_action


def compute_ik_rel_action(
    gripper_position: float,
    include_gripper: bool,
    gripper_threshold: float,
    leader_interface,
) -> np.ndarray:
    """Convert successive leader EE poses into an IK-relative delta action.

    Computes the difference between the current and previous cartesian pose
    from the driver's FK, then divides by the action scale.  On the first
    call the delta is zero (no previous pose to diff against).
    """
    robot_output = leader_interface.driver.get_robot_output()
    cartesian = np.array(robot_output.cartesian.positions)
    delta = (cartesian - _last_cartesian.get("pos", cartesian)) / IK_REL_ACTION_SCALE
    _last_cartesian["pos"] = cartesian.copy()

    if include_gripper:
        gripper_action = 1.0 if gripper_position > gripper_threshold else -1.0
        action = np.concatenate([delta, [gripper_action]])
    else:
        action = delta

    return action


def main() -> None:
    action_type = detect_action_type(args_cli.task)
    gripper = has_gripper_action(args_cli.task)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.env_name = args_cli.task

    # Disable timeout for teleoperation (run indefinitely)
    env_cfg.terminations.time_out = None

    # For lift tasks, add goal-reached termination
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(
            func=mdp.object_reached_goal
        )

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    leader: LeaderArmHardware
    leader = LeaderArmHardware(ip=args_cli.leader_ip)

    try:
        leader.connect()
    except Exception as e:
        logger.error(f"Failed to connect to leader arm: {e}")
        env.close()
        simulation_app.close()
        return

    env.reset()
    print("Teleoperation active. Ctrl+C to stop.")

    step = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Read leader arm state
                arm_positions, gripper_position = leader.get_state()
                gripper_position = max(0.0, min(0.044, gripper_position))

                # Compute action based on environment's action type
                if action_type == "joint_pos":
                    action_np = compute_joint_pos_action(
                        arm_positions,
                        gripper_position,
                        gripper,
                        args_cli.gripper_threshold,
                    )
                elif action_type == "ik_abs":
                    action_np = compute_ik_abs_action(
                        gripper_position,
                        gripper,
                        args_cli.gripper_threshold,
                        leader,
                    )
                elif action_type == "ik_rel":
                    action_np = compute_ik_rel_action(
                        gripper_position,
                        gripper,
                        args_cli.gripper_threshold,
                        leader,
                    )
                else:
                    raise ValueError(f"Unknown action type: {action_type}")

                # Convert to torch and apply to all environments
                action = torch.tensor(
                    action_np, dtype=torch.float32, device=env.device
                ).unsqueeze(0)
                actions = action.repeat(env.num_envs, 1)

                # Step the environment
                env.step(actions)

                step += 1
                if step % 60 == 0:
                    joints_str = "  ".join(
                        [
                            f"J{i}:{np.degrees(p):+6.1f}\u00b0"
                            for i, p in enumerate(arm_positions)
                        ]
                    )
                    grip_str = f"Grip:{'OPEN' if gripper_position > args_cli.gripper_threshold else 'CLOSE'}"
                    print(
                        f"\r[Step {step:6d}] {joints_str}  {grip_str}    ",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        pass
    finally:
        leader.cleanup()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

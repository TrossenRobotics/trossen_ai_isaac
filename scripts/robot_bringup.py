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
Robot Bringup Script for Isaac Sim

This script loads a Trossen robot model into Isaac Sim with a ground plane.

Usage:
    ~/isaacsim5.1/isaac-sim.sh scripts/robot_bringup.py [robot_name]

Examples:
    ~/isaacsim5.1/isaac-sim.sh scripts/robot_bringup.py mobile_ai
    ~/isaacsim5.1/isaac-sim.sh scripts/robot_bringup.py stationary_ai
    ~/isaacsim5.1/isaac-sim.sh scripts/robot_bringup.py wxai_base

Available Robots:
    - mobile_ai: Dual-arm mobile manipulator
    - stationary_ai: Dual-arm stationary platform
    - wxai_base: Single arm base configuration
    - wxai_follower: Single arm follower configuration
    - wxai_leader_left: Left leader arm
    - wxai_leader_right: Right leader arm
"""

import os
import sys

from isaacsim import SimulationApp

# Must initialize SimulationApp before importing other Isaac Sim modules
simulation_app = SimulationApp({"headless": False})

import carb
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

# List of available robot models
available_robots = [
    "mobile_ai",
    "stationary_ai",
    "wxai_base",
    "wxai_follower",
    "wxai_leader_left",
    "wxai_leader_right",
]

default_robot = "mobile_ai"
if len(sys.argv) > 1:
    robot_name = sys.argv[1]
else:
    robot_name = default_robot
    carb.log_warn(f"No robot specified. Using default: {robot_name}")

if robot_name not in available_robots:
    carb.log_error(f"Invalid robot name: '{robot_name}'")
    carb.log_error(f"Available robots: {', '.join(available_robots)}")
    simulation_app.close()
    exit()

asset_path = f"./assets/robots/{robot_name}.usd"
prim_path = f"/{robot_name}"

carb.log_warn(f"Loading robot: {robot_name}")

# Check if the robot file exists
if not os.path.exists(asset_path):
    carb.log_error(f"Could not find USD file at: {asset_path}")
    simulation_app.close()
    exit()

# Create the simulation world
my_world = World(stage_units_in_meters=1.0)

my_world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)

my_world.reset()

while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()
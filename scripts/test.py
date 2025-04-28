import os
from pathlib import Path
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import h5py

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrim
import omni.usd

from modules.trossen_arm_controller import TrossenArmController
from isaacsim.core.utils.types import ArticulationAction
from trossen_arm_utils import *
import global_var
from isaacsim.core.prims import SingleArticulation

folder_dir = str(Path(__file__).parent.parent.resolve())

# Configuration Paths
USD_PATH = "/home/shuhang/isaac_ws/trossen_ai_isaac/trossen_ai_arm_usd/wxai_follower.usd"
LEFT_ARM_PATH = "/World/wxai"

omni.usd.get_context().open_stage(USD_PATH, None)
world = World(stage_units_in_meters=1.0)
world.reset()

prim = SingleArticulation(prim_path=LEFT_ARM_PATH, name="aloha_arm")
prim.initialize()
action = ArticulationAction(joint_positions=np.array([0.0, np.pi/12, np.pi/12, 0.0, 0.0, 0.0]), joint_indices=np.array([0, 1, 2, 3, 4, 5]))
prim.apply_action(action)

for i in range(2000):
    world.step(render=True)




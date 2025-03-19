"""
Isaac Sim Dual-Arm Transfer Simulation

This script sets up a simulated environment in Isaac Sim where two robotic arms collaborate to pick up 
and transfer a box. The simulation records camera feeds and robot states, saving them in an HDF5 file 
and an AVI video.

Usage:
    source ~/isaacsim/setup_conda_env.sh (If use conda)
    source ~/isaacsim/setup_python_env.sh  (If use system python)
    python scripts/main.py

Requirements:
    - NVIDIA Isaac Sim
    - OpenCV (`cv2`)
    - NumPy (`numpy`)
    - h5py
    - Matplotlib (`matplotlib`)

Key Features:
    - Initializes a simulation world with two robotic arms.
    - Captures and records multiple camera views.
    - Saves robot states (joint positions, velocities, torques) in an HDF5 file.
    - Performs a pick-and-place task with dynamic box placement.
    - Handles a robotic handover between two arms.

Ensure Isaac Sim is installed and all dependencies are available before running the script.
"""
import os
from pathlib import Path
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import h5py

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrim
import omni.usd

from modules.trossen_arm_controller import TrossenArmController
from trossen_arm_utils import *
import global_var

folder_dir = str(Path(__file__).parent.parent.resolve())

# Configuration Paths
USD_PATH = folder_dir + "/trossen_ai_scene/trossen_ai_scene.usd"
LEFT_ARM_PATH = "/World/left_arm"
RIGHT_ARM_PATH = "/World/right_arm"
LULA_DESC_PATH = folder_dir + "/trossen_ai_scene/trossen_ai_arm.yaml"
LULA_URDF_PATH = folder_dir + "/trossen_ai_scene/wxai_follower.urdf"
CAMERA_PRIM_PATHS = [
    "/World/cam_high",
    "/World/cam_low",
    "/World/left_arm/link_6/camera_mount_d405/cam_left_arm",
    "/World/right_arm/link_6/camera_mount_d405/cam_right_arm"
]
BOX_PRIM_PATH = "/World/aloha_scene_joint/box/box"

def main(args):
    on_screen_render = args.onscreen_render if args.onscreen_render else False
    recording = args.recording if args.recording else False

    # Initialize Simulation
    omni.usd.get_context().open_stage(USD_PATH, None)
    cameras = [Camera(prim_path=path, resolution=(640, 480)) for path in CAMERA_PRIM_PATHS]

    world = World(stage_units_in_meters=1.0)
    world.reset()
    box_prim = XFormPrim(prim_path=BOX_PRIM_PATH)

    frame_width, frame_height = 1280, 960  # 2x2 grid (640x480 each)
    fps = 30
    output_filename = "output_video.avi"
    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

    # Initialize camera and video writer
    if recording or on_screen_render:
        for camera in cameras:
            camera.initialize()
            camera.add_motion_vectors_to_frame()
        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    else:
        axs = None

    if recording:
        # Create directory for HDF5 storage
        os.makedirs("output_dataset", exist_ok=True)
        dataset_path = os.path.join("output_dataset", "episode_1.hdf5")

        # Open HDF5 file
        hdf5_file = h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2)
        hdf5_file.attrs['sim'] = True

        # Create HDF5 groups
        obs_grp = hdf5_file.create_group('observations')
        image_grp = obs_grp.create_group('images')

        for cam_name in CAMERA_PRIM_PATHS:
            image_grp.create_dataset(cam_name, shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype='uint8')

        # Create datasets for robot states
        for arm in ['left_arm', 'right_arm']:
            obs_grp.create_dataset(f"{arm}/qpos", shape=(0, 8), maxshape=(None, 8), dtype='float64')
            obs_grp.create_dataset(f"{arm}/qvel", shape=(0, 8), maxshape=(None, 8), dtype='float64')
            obs_grp.create_dataset(f"{arm}/qtorque", shape=(0, 8, 3), maxshape=(None, 8, 3), dtype='float64')
    else:
        obs_grp = None
        image_grp = None

    # Initialize Arms
    left_arm = TrossenArmController(LEFT_ARM_PATH, "ee_gripper_link", "left_arm_robot", LULA_DESC_PATH, LULA_URDF_PATH)
    right_arm = TrossenArmController(RIGHT_ARM_PATH, "ee_gripper_link", "right_arm_robot", LULA_DESC_PATH, LULA_URDF_PATH)

    global_var.set_shared_value(world, cameras, video_writer, recording, axs, on_screen_render, left_arm, right_arm, obs_grp, image_grp, CAMERA_PRIM_PATHS)

    for _ in range(100):
        world.step(render=True)

    left_arm.initialize()
    right_arm.initialize()

    # Randomize the location of the box
    box_pos, box_orient = randomize_box_pose(box_prim, position_range=0.2, z_height=0.02)
    box_yaw, _ = quaternion_to_yaw(box_orient)

    # Decide which arm to act first based on their distance with the box 
    if box_pos[0] < 0:
        # Closer to left arm
        # Left arm will pick up the box and move to the center of the stage
        execute_pick_and_place(left_arm, box_pos[0], box_pos[1], box_yaw)
        box_position, _ = box_prim.get_world_pose()
        # Right arm's end effector will rotate 90 degree, go and take box from the left arm
        handover_and_place(left_arm, right_arm, box_position, 0.044, 0.001)
    else:
        # Right arm will pick up the box and move to the center of the stage
        execute_pick_and_place(right_arm, box_pos[0], box_pos[1], box_yaw, invert_y=True)
        box_position, _ = box_prim.get_world_pose()
        # Left arm's end effector will rotate 90 degree, go and take box from the right arm
        handover_and_place(right_arm, left_arm, box_position, 0.044, 0.001)

    if recording or on_screen_render:
        # If we save videos locally, the recording will stop when both arm go to the end position
        for _ in range(120):
            capture_and_save_frames()
        if hdf5_file:
            hdf5_file.flush()  # Ensures all data is written before closing
            hdf5_file.close()
            print("HDF5 file properly closed.")
        else:
            print("Failed")
        simulation_app.close()
    else:
        # Else keep rendering until user press any key or use ctrl-C
        try:
            while True:
                capture_and_save_frames()
        except KeyboardInterrupt:
            print("Simulation ended. Closing video writer.")
            if recording:
                video_writer.release()
            simulation_app.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', action='store_true', help='save file locally')
    parser.add_argument('--onscreen_render', action='store_true', help='real-time render')

    args = parser.parse_args()
    main(args)
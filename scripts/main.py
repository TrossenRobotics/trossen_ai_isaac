from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
import omni.usd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from trossen_arm_controller import TrossenArmController
from trossen_arm_utils import *
from omni.isaac.core.prims import XFormPrim
from pathlib import Path
folder_dir = str(Path(__file__).parent.parent.resolve())
print(folder_dir)

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

video_writer = None

# Initialize Simulation
omni.usd.get_context().open_stage(USD_PATH, None)
cameras = [Camera(prim_path=path, resolution=(640, 480)) for path in CAMERA_PRIM_PATHS]

world = World(stage_units_in_meters=1.0)
world.reset()
box_prim = XFormPrim(prim_path=BOX_PRIM_PATH)

if save_local:
    for camera in cameras:
        camera.initialize()
        camera.add_motion_vectors_to_frame()

    frame_width, frame_height = 1280, 960  # 2x2 grid (640x480 each)
    fps = 30
    output_filename = "output_video.avi"
    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# Initialize Arms
left_arm = TrossenArmController(world, cameras, LEFT_ARM_PATH, "ee_gripper_link", "left_arm_robot", LULA_DESC_PATH, LULA_URDF_PATH, video_writer)
right_arm = TrossenArmController(world, cameras, RIGHT_ARM_PATH, "ee_gripper_link", "right_arm_robot", LULA_DESC_PATH, LULA_URDF_PATH, video_writer)

for _ in range(20):
    world.step(render=True)

left_arm.initialize()
right_arm.initialize()

arm_start_pos, _, _ = left_arm.get_current_pos_estimation()

for _ in range(20):
    world.step(render=True)

left_arm.get_current_pos_estimation()
box_pos, box_orient = randomize_box_pose(box_prim, position_range=0.2, z_height=0.02)
box_yaw, _ = quaternion_to_yaw(box_orient)

# Main execution logic
perform_initial_steps(world)

if box_pos[0] < 0:
    execute_pick_and_place(world, left_arm, box_pos[0], box_pos[1], box_yaw)
    box_position, _ = box_prim.get_world_pose()
    handover_and_place(world, left_arm, right_arm, box_position, 0.044, 0.001)
else:
    execute_pick_and_place(world, right_arm, box_pos[0], box_pos[1], box_yaw, invert_y=True)
    box_position, _ = box_prim.get_world_pose()
    handover_and_place(world, right_arm, left_arm, box_position, 0.044, 0.001)

if save_local:
    for _ in range(120):
        capture_and_save_frames(world, cameras, video_writer)
    simulation_app.close()
else:
    try:
        while True:
            capture_and_save_frames(world, cameras)
    except KeyboardInterrupt:
        print("Simulation ended. Closing video writer.")
        if save_local:
            video_writer.release()
            simulation_app.close()

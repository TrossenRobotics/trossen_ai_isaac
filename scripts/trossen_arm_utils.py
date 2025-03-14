import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import cv2

save_local = True
on_screen_render = False

def euler_to_quaternion(roll, pitch, yaw):
    rotation = R.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)])
    return np.roll(rotation.as_quat(), shift=1)

def quaternion_to_yaw(quat):
    rotation = R.from_quat(np.roll(quat, shift=-1))
    euler = rotation.as_euler('xyz', degrees=True)
    return euler[2], euler

def randomize_box_pose(box, position_range=0.1, z_height=0.02):
    while True:
        random_x = random.uniform(-position_range, position_range)
        random_y = random.uniform(-position_range, position_range)
        box_position = np.array([random_x, random_y, z_height])
        random_yaw = random.uniform(0, 90)
        box_orientation = euler_to_quaternion(0, 0, random_yaw)
        if np.linalg.norm(box_position - (0.469, 0, 0)) <= 0.469:
            break
    box.set_world_pose(box_position, box_orientation)
    return box_position, box_orientation

def perform_initial_steps(world):
    for _ in range(100):
        world.step(render=True)

def move_arm_to_target(arm, start_pos, goal_pos, start_orient, goal_orient, frame):
    arm.move_to_target(start_pos, goal_pos, start_orientation=start_orient, goal_orientation=goal_orient, frame=frame)

def execute_pick_and_place(world, arm, box_x, box_y, box_yaw, invert_y=False):
    sign = -1 if invert_y else 1

    arm.move_to_target(
        start_pos = arm.arm_start_pos,
        goal_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(0, 90, 0),
        frame = "ee_gripper_link"
    )

    arm.move_to_target(
        start_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        goal_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        start_orientation = euler_to_quaternion(0, 90, 0),
        goal_orientation = euler_to_quaternion(0, 90, box_yaw),
        frame = "ee_gripper_link"
    )
    
    arm.move_to_target(
        start_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        goal_pos = [0.469 + sign * box_x, sign * box_y, 0.04],
        start_orientation = euler_to_quaternion(0, 90, box_yaw),
        goal_orientation = euler_to_quaternion(0, 90, box_yaw),
        frame = "ee_gripper_link"
    )

    arm.apply_grasp(0.001)

    arm.move_to_target(
        start_pos = [0.469 + sign * box_x, sign * box_y, 0.04],
        goal_pos = [0.469, 0, 0.25],
        start_orientation = euler_to_quaternion(0, 90, box_yaw),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame = "ee_gripper_link"
    )

def handover_and_place(world, first_arm, second_arm, box_position, first_arm_grasp, second_arm_grasp):
    second_arm.move_to_target(
        start_pos = second_arm.arm_start_pos,
        goal_pos = [0.469, 0, box_position[2] + 0.02],
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(90, 0, 0),
        frame="ee_gripper_link"
    )

    second_arm.apply_grasp(second_arm_grasp)

    first_arm.apply_grasp(first_arm_grasp)

    second_arm.move_to_target(
        start_pos = [0.469, 0, box_position[2] + 0.02],
        goal_pos = second_arm.arm_start_pos,
        start_orientation = euler_to_quaternion(90, 0, 0),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame="ee_gripper_link"
    )

    first_arm.move_to_target(
        start_pos = [0.469, 0, 0.25],
        goal_pos = first_arm.arm_start_pos,
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame="ee_gripper_link"
    )

def capture_and_save_frames(world, cameras, video_writer=None):
    world.step(render=True)
    if save_local:
        combined_frame = np.zeros((960, 1280, 3), dtype=np.uint8)  # 2x2 grid (640x480 each)
        
        for idx, cam in enumerate(cameras):
            rgba_frame = cam.get_rgba()
            bgr_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)

            row, col = divmod(idx, 2)
            start_y, start_x = row * 480, col * 640
            combined_frame[start_y:start_y+480, start_x:start_x+640] = bgr_frame

        video_writer.write(combined_frame)
        
        if on_screen_render:
            for ax in axs.flatten():
                ax.clear()
            axs[0, 0].imshow(combined_frame[:480, :640, ::-1])  # Top-left
            axs[0, 1].imshow(combined_frame[:480, 640:, ::-1])  # Top-right
            axs[1, 0].imshow(combined_frame[480:, :640, ::-1])  # Bottom-left
            axs[1, 1].imshow(combined_frame[480:, 640:, ::-1])  # Bottom-right
            for ax in axs.flatten():
                ax.axis('off')
            plt.draw()
            plt.pause(0.01) 

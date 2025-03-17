import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import cv2
import global_var

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a quaternion.

    Args:
        roll (float): Rotation around the X-axis in degrees.
        pitch (float): Rotation around the Y-axis in degrees.
        yaw (float): Rotation around the Z-axis in degrees.

    Returns:
        np.ndarray: Quaternion representation (w, x, y, z).
    """
    rotation = R.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)])
    return np.roll(rotation.as_quat(), shift=1)

def quaternion_to_yaw(quat):
    """
    Extracts the yaw (rotation around the Z-axis) from a quaternion.

    Args:
        quat (np.ndarray): Quaternion (w, x, y, z).

    Returns:
        tuple: Yaw angle in degrees and the full Euler angles (roll, pitch, yaw).
    """
    rotation = R.from_quat(np.roll(quat, shift=-1))
    euler = rotation.as_euler('xyz', degrees=True)
    return euler[2], euler

def randomize_box_pose(box, position_range=0.1, z_height=0.02):
    """
    Randomizes the position and orientation of a box within a given range.

    Args:
        box (object): The box object to set position and orientation.
        position_range (float): Range limit for X and Y displacement.
        z_height (float): Fixed Z-height for the box.

    Returns:
        tuple: New position and orientation (quaternion) of the box.
    """
    while True:
        random_x = random.uniform(-position_range, position_range)
        random_y = random.uniform(-position_range, position_range)
        box_position = np.array([random_x, random_y, z_height])
        random_yaw = random.uniform(0, 90)
        box_orientation = euler_to_quaternion(0, 0, random_yaw)
        if np.linalg.norm(box_position - (0.469, 0, 0)) <= 0.469 or np.linalg.norm(box_position - (-0.469, 0, 0)) <= 0.469:
            break
    print("Box position: ")
    print(box_position)
    box.set_world_pose(box_position, box_orientation)
    for _ in range(100):
        capture_and_save_frames()
    return box_position, box_orientation

def execute_pick_and_place(arm, box_x, box_y, box_yaw, invert_y=False):
    """
    Executes a pick-and-place operation for an arm to grasp and relocate a box.

    Args:
        world (object): The simulation world object.
        arm (object): The robotic arm controller instance.
        box_x (float): X-coordinate of the box.
        box_y (float): Y-coordinate of the box.
        box_yaw (float): Orientation (yaw) of the box.
        invert_y (bool): If True, flips the Y-axis movement direction.
    """
    sign = -1 if invert_y else 1

    # arm 1 will move to the top of the box
    arm.move_to_target(
        start_pos = arm.arm_start_pos,
        goal_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(0, 90, 0),
        frame = "ee_gripper_link"
    )

    # arm 1 rotate to have the same orientation as the box
    arm.move_to_target(
        start_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        goal_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        start_orientation = euler_to_quaternion(0, 90, 0),
        goal_orientation = euler_to_quaternion(0, 90, box_yaw),
        frame = "ee_gripper_link"
    )
    
    # arm 1 moves down
    arm.move_to_target(
        start_pos = [0.469 + sign * box_x, sign * box_y, 0.10],
        goal_pos = [0.469 + sign * box_x, sign * box_y, 0.04],
        start_orientation = euler_to_quaternion(0, 90, box_yaw),
        goal_orientation = euler_to_quaternion(0, 90, box_yaw),
        frame = "ee_gripper_link"
    )

    # arm 1 grasps the box
    arm.apply_grasp(0.001)

    # arm 1 lifts the box and move to the center of the stage
    arm.move_to_target(
        start_pos = [0.469 + sign * box_x, sign * box_y, 0.04],
        goal_pos = [0.469, 0, 0.25],
        start_orientation = euler_to_quaternion(0, 90, box_yaw),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame = "ee_gripper_link"
    )

def handover_and_place(first_arm, second_arm, box_position, first_arm_grasp, second_arm_grasp):
    """
    Handles object handover between two robotic arms and places it in the final location.

    Args:
        world (object): The simulation world object.
        first_arm (object): First arm that initially holds the object.
        second_arm (object): Second arm that takes the object.
        box_position (list): Final position of the box [x, y, z].
        first_arm_grasp (float): Grip state for the first arm.
        second_arm_grasp (float): Grip state for the second arm.
    """

    # arm 2 moves to the center of the statge and at the same time rotate 90 degree
    second_arm.move_to_target(
        start_pos = second_arm.arm_start_pos,
        goal_pos = [0.469, 0, box_position[2] + 0.02],
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(90, 0, 0),
        frame="ee_gripper_link"
    )

    # arm 2 grasps the box
    second_arm.apply_grasp(second_arm_grasp)

    # arm 1 releases the box
    first_arm.apply_grasp(first_arm_grasp)

    # arm 2 takes the box to the final position
    second_arm.move_to_target(
        start_pos = [0.469, 0, box_position[2] + 0.02],
        goal_pos = second_arm.arm_start_pos,
        start_orientation = euler_to_quaternion(90, 0, 0),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame="ee_gripper_link"
    )

    # arm 1 moves to the final position
    first_arm.move_to_target(
        start_pos = [0.469, 0, 0.25],
        goal_pos = first_arm.arm_start_pos,
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame="ee_gripper_link"
    )

def capture_and_save_frames():
    """
    Captures frames from multiple cameras and saves them to a video file.

    Args:
        world (object): The simulation world object.
        cameras (list): List of camera objects.
        video_writer (cv2.VideoWriter, optional): Video writer object for saving frames.
    """
    global_var.shared_world.step(render=True)
    if global_var.shared_recording:
        combined_frame = np.zeros((960, 1280, 3), dtype=np.uint8)  # 2x2 grid (640x480 each)
        
        for idx, cam in enumerate(global_var.shared_cameras):
            rgba_frame = cam.get_rgba()
            bgr_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)

            row, col = divmod(idx, 2)
            start_y, start_x = row * 480, col * 640
            combined_frame[start_y:start_y+480, start_x:start_x+640] = bgr_frame

        global_var.shared_video_writer.write(combined_frame)
        
        if global_var.shared_on_screen_render:
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

"""
Robotic Arm Task Execution and Data Capture

This module provides functions for robotic arm operations in Isaac Sim, including motion planning, 
grasping, and handover between two arms. It also handles data recording, capturing images, and 
saving robot states for further analysis.

Functions:
    - euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray
        Converts Euler angles (roll, pitch, yaw) to a quaternion.
    - quaternion_to_yaw(quat: np.ndarray) -> tuple[float, np.ndarray]
        Extracts the yaw angle from a quaternion.
    - randomize_box_pose(box, position_range: float = 0.1, z_height: float = 0.02) -> tuple[np.ndarray, np.ndarray]
        Randomizes the position and orientation of a box within a defined range.
    - execute_pick_and_place(arm, box_x: float, box_y: float, box_yaw: float, left_arm_first: bool = False) -> None
        Executes a pick-and-place operation with a robotic arm.
    - handover_and_place(first_arm, second_arm, box_position: np.ndarray) -> None
        Handles object handover between two robotic arms and places it in a final location.
    - capture_and_save_frames() -> None
        Captures frames from multiple cameras and saves robot state data to an HDF5 file.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import random
import cv2
import matplotlib.pyplot as plt

import global_var

left_arm_pos = [-0.4575, -0.019, 0.02]
right_arm_pos = [0.4575, -0.019, 0.02]

def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    :param roll: Rotation around the X-axis in degrees.
    :type roll: float
    :param pitch: Rotation around the Y-axis in degrees.
    :type pitch: float
    :param yaw: Rotation around the Z-axis in degrees.
    :type yaw: float
    :return: Quaternion representation (w, x, y, z).
    :rtype: np.ndarray
    """
    rotation = R.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)])
    return np.roll(rotation.as_quat(), shift=1)

def quaternion_to_yaw(quat: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Extract the yaw (rotation around the Z-axis) from a quaternion.

    :param quat: Quaternion (w, x, y, z).
    :type quat: np.ndarray
    :return: Yaw angle in degrees and the full Euler angles (roll, pitch, yaw).
    :rtype: tuple[float, np.ndarray]
    """
    rotation = R.from_quat(np.roll(quat, shift=-1))
    euler = rotation.as_euler('xyz', degrees=True)
    return euler[2], euler

def randomize_box_pose(box, position_range: float = 0.1, z_height: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomize the position and orientation of a box within a given range.

    :param box: The box object to set position and orientation.
    :type box: object
    :param position_range: Range limit for X and Y displacement.
    :type position_range: float
    :param z_height: Fixed Z-height for the box.
    :type z_height: float
    :return: New position and orientation (quaternion) of the box.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    while True:
        random_x = -1 * abs(random.uniform(-position_range, position_range))
        random_y = random.uniform(-position_range, position_range)
        box_position = np.array([random_x, random_y, z_height])
        # box_position = np.array([-0.1, 0, z_height])  # Set Y to 0 for simplicity
        random_yaw = random.uniform(0, 90)
        box_orientation = euler_to_quaternion(0, 0, random_yaw)
        if np.linalg.norm(box_position - (0.4575, -0.019, 0.02)) <= 0.4575 or np.linalg.norm(box_position - (-0.4575, -0.019, 0.02)) <= 0.4575:
            break
    print("Box position: ")
    print(box_position)
    print("Box yaw: ")
    print(random_yaw)
    box.set_world_pose(box_position, box_orientation)
    # for _ in range(100):
    #     capture_and_save_frames()
    return box_position, box_orientation

def world_to_arm_frame(goal_x, goal_y, goal_z, is_left_arm):
    """
    Convert the goal position from world coordinates to arm frame coordinates.
    :param goal_x: X-coordinate of the goal position.
    :type goal_x: float
    :param goal_y: Y-coordinate of the goal position.
    :type goal_y: float
    :param goal_z: Z-coordinate of the goal position.
    :type goal_z: float
    :param is_left_arm: Boolean indicating if the left arm is being used.
    :type is_left_arm: bool
    :return: Transformed position in the arm's coordinate frame.
    :rtype: list[float]
    """
    if is_left_arm:
        return [-left_arm_pos[0] + goal_x, -left_arm_pos[1] + goal_y, goal_z - left_arm_pos[2]]
    else:
        return [right_arm_pos[0] - goal_x, right_arm_pos[1] - goal_y, goal_z - right_arm_pos[2]]

def execute_pick_and_place(arm, box_x: float, box_y: float, box_yaw: float, left_arm_first: bool = True) -> None:
    """
    Execute a pick-and-place operation for an arm to grasp and relocate a box.

    :param arm: The robotic arm controller instance.
    :type arm: object
    :param box_x: X-coordinate of the box.
    :type box_x: float
    :param box_y: Y-coordinate of the box.
    :type box_y: float
    :param box_yaw: Orientation (yaw) of the box.
    :type box_yaw: float
    :param left_arm_first: If True, flips the Y-axis movement direction.
    :type left_arm_first: bool
    """

    # arm 1 will move to the top of the box
    arm.set_ee_pos(
        start_pos = arm.arm_start_pos,
        goal_pos = world_to_arm_frame(box_x, box_y, 0.10, left_arm_first),
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(0, 90, 0),
        frame = "ee_gripper_link"
    )

    # arm 1 rotate to have the same orientation as the box
    arm.set_ee_pos(
        start_pos = world_to_arm_frame(box_x, box_y, 0.10, left_arm_first),
        goal_pos = world_to_arm_frame(box_x, box_y, 0.10, left_arm_first),
        start_orientation = euler_to_quaternion(0, 90, 0),
        goal_orientation = euler_to_quaternion(0, 90, box_yaw),
        frame = "ee_gripper_link"
    )
    
    # arm 1 moves down
    arm.set_ee_pos(
        start_pos = world_to_arm_frame(box_x, box_y, 0.10, left_arm_first),
        goal_pos = world_to_arm_frame(box_x, box_y, 0.02, left_arm_first),
        start_orientation = euler_to_quaternion(0, 90, box_yaw),
        goal_orientation = euler_to_quaternion(0, 90, box_yaw),
        frame = "ee_gripper_link"
    )

    # arm 1 grasps the box
    arm.grasp()

    # arm 1 lifts the box and move to the center of the stage
    arm.set_ee_pos(
        start_pos = world_to_arm_frame(box_x, box_y, 0.02, left_arm_first),
        goal_pos = world_to_arm_frame(0, 0, 0.25, left_arm_first),
        start_orientation = euler_to_quaternion(0, 90, box_yaw),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame = "ee_gripper_link"
    )

def handover_and_place(first_arm, second_arm, box_position: np.ndarray, left_arm_first: bool = True) -> None:
    """
    Handles object handover between two robotic arms and places it in the final location.

    :param first_arm: First arm that initially holds the object.
    :type first_arm: object
    :param second_arm: Second arm that takes the object.
    :type second_arm: object
    :param box_position: Final position of the box [x, y, z].
    :type box_position: np.ndarray
    """
    
    # arm 2 moves to the center of the statge and at the same time rotate 90 degree
    second_arm.set_ee_pos(
        start_pos = second_arm.arm_start_pos,
        goal_pos = world_to_arm_frame(box_position[0], box_position[1], box_position[2], not left_arm_first),
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(90, 0, 0),
        frame="ee_gripper_link"
    )

    # arm 2 grasps the box
    second_arm.grasp()

    # arm 1 releases the box
    first_arm.release()

    # arm 2 takes the box to the final position
    second_arm.set_ee_pos(
        start_pos = world_to_arm_frame(box_position[0], box_position[1], box_position[2], not left_arm_first),
        goal_pos = second_arm.arm_start_pos,
        start_orientation = euler_to_quaternion(90, 0, 0),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame="ee_gripper_link"
    )

    # arm 1 moves to the final position
    first_arm.set_ee_pos(
        start_pos = world_to_arm_frame(0, 0, 0.25, left_arm_first),
        goal_pos = first_arm.arm_start_pos,
        start_orientation = euler_to_quaternion(0, 0, 0),
        goal_orientation = euler_to_quaternion(0, 0, 0),
        frame="ee_gripper_link"
    )

def capture_and_save_frames() -> None:
    """
    Capture frames from multiple cameras and save them to a video file.
    """
    global_var.shared_world.step(render=True)
    if global_var.shared_recording:
        left_arm = global_var.shared_left_arm
        right_arm = global_var.shared_right_arm

        # Get robot data
        qpos_left = left_arm.get_current_joint_positions()
        qpos_right = right_arm.get_current_joint_positions()
        qvel_left = left_arm.get_current_joint_velocities()
        qvel_right = right_arm.get_current_joint_velocities()
        qtorque_left = left_arm.get_current_joint_torques()
        qtorque_right = right_arm.get_current_joint_torques()

        # Update HDF5 datasets
        def append_to_dataset(dataset, data):
            dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
            dataset[-1] = data
        
        # Append data to hdf5
        append_to_dataset(global_var.shared_obs_grp[f'left_arm/qpos'], qpos_left)
        append_to_dataset(global_var.shared_obs_grp[f'right_arm/qpos'], qpos_right)
        append_to_dataset(global_var.shared_obs_grp[f'left_arm/qvel'], qvel_left)
        append_to_dataset(global_var.shared_obs_grp[f'right_arm/qvel'], qvel_right)
        append_to_dataset(global_var.shared_obs_grp[f'left_arm/qtorque'], qtorque_left)
        append_to_dataset(global_var.shared_obs_grp[f'right_arm/qtorque'], qtorque_right)
    
    if global_var.shared_recording or global_var.shared_on_screen_render:
        # Process the images
        combined_frame = np.zeros((960, 1280, 3), dtype=np.uint8)  # 2x2 grid (640x480 each)
        for idx, cam in enumerate(global_var.shared_cameras):
            rgba_frame = cam.get_rgba()
            rgb_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2RGB)
            bgr_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)

            if global_var.shared_recording:
                append_to_dataset(global_var.shared_image_grp[global_var.shared_camera_list[idx]], rgb_frame)

            row, col = divmod(idx, 2)
            start_y, start_x = row * 480, col * 640
            combined_frame[start_y:start_y+480, start_x:start_x+640] = bgr_frame
        
        if global_var.shared_recording:
            global_var.shared_video_writer.write(combined_frame)
        
        if global_var.shared_on_screen_render:
            for ax in global_var.shared_axs.flatten():
                ax.clear()
            global_var.shared_axs[0, 0].imshow(combined_frame[:480, :640, ::-1])  # Top-left
            global_var.shared_axs[0, 1].imshow(combined_frame[:480, 640:, ::-1])  # Top-right
            global_var.shared_axs[1, 0].imshow(combined_frame[480:, :640, ::-1])  # Bottom-left
            global_var.shared_axs[1, 1].imshow(combined_frame[480:, 640:, ::-1])  # Bottom-right
            for ax in global_var.shared_axs.flatten():
                ax.axis('off')
            plt.draw()
            plt.pause(0.01) 

def slerp(start_quat: list[float], end_quat: list[float], t_values: np.ndarray) -> np.ndarray:
    """
    Perform spherical linear interpolation (SLERP) between two quaternions.

    :param start_quat: Starting quaternion (w, x, y, z).
    :type start_quat: list[float]
    :param end_quat: Target quaternion (w, x, y, z).
    :type end_quat: list[float]
    :param t_values: Array of interpolation time steps (0 to 1).
    :type t_values: np.ndarray
    :return: Interpolated quaternion sequence.
    :rtype: np.ndarray
    """
    key_times = [0, 1]
    key_rots = R.from_quat([start_quat, end_quat])
    slerp_interpolator = Slerp(key_times, key_rots)
    interpolated_rots = slerp_interpolator(t_values)
    return interpolated_rots.as_quat()

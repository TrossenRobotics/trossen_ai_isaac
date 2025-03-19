import os

import h5py
import cv2
import numpy as np

# Path to the HDF5 file
hdf5_path = "output_dataset/episode_1.hdf5"
output_video = "combined_output.avi"

# Video settings
frame_width, frame_height = 1280, 960  # 2x2 grid (each image 640x480)
fps = 30  # Frames per second

# Open HDF5 file
with h5py.File(hdf5_path, 'r') as hdf5_file:
    # Camera names (ensure they match your dataset)
    camera_list = [
        "World/cam_high",
        "World/cam_low",
        "World/left_arm/link_6/camera_mount_d405/cam_left_arm",
        "World/right_arm/link_6/camera_mount_d405/cam_right_arm"
    ]

    # Check if all camera datasets exist
    for cam in camera_list:
        if cam not in hdf5_file:
            raise ValueError(f"Missing dataset in HDF5: {cam}")

    # Get the number of frames
    num_frames = hdf5_file[camera_list[0]].shape[0]
    print(f"Total frames: {num_frames}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    for i in range(num_frames):
        imgs = []
        
        # Read images from HDF5 for each camera
        for cam in camera_list:
            img = hdf5_file[cam][i]  # Get image frame
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
            imgs.append(img)

        # Combine images into a 2x2 grid
        top_row = np.hstack((imgs[0], imgs[1]))
        bottom_row = np.hstack((imgs[2], imgs[3]))
        combined_frame = np.vstack((top_row, bottom_row))

        # Write frame to video
        video_writer.write(combined_frame)

    # Release video writer
    video_writer.release()
    print(f"Video saved as {output_video}")

"""
Global Shared Variables for Simulation

This module maintains shared global variables used in the simulation, including world state, cameras, 
video recording settings, robotic arms, and observation data groups. 

Usage:
    Call `set_shared_value()` to initialize these variables before running the simulation.
"""

shared_world = None
shared_cameras = None
shared_video_writer = None
shared_recording = None
shared_on_screen_render = None
shared_left_arm = None
shared_right_arm = None
shared_obs_grp = None
shared_image_grp = None
shared_camera_list = None
shared_axs = None

def set_shared_value(world, cameras, video_writer, recording, axs, on_screen_render, left_arm, right_arm, obs_grp, image_grp, camera_list):
    global shared_world, shared_cameras, shared_video_writer, shared_recording, shared_on_screen_render, shared_left_arm, shared_right_arm, shared_obs_grp, shared_image_grp, shared_camera_list, shared_axs
    shared_world = world
    shared_cameras = cameras
    shared_video_writer = video_writer
    shared_recording = recording
    shared_on_screen_render = on_screen_render
    shared_left_arm = left_arm
    shared_right_arm = right_arm
    shared_obs_grp = obs_grp
    shared_image_grp = image_grp
    shared_camera_list = camera_list
    shared_axs = axs
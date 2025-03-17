shared_world = None
shared_cameras = None
shared_video_writer = None
shared_recording = None
shared_on_screen_render = None

def set_shared_value(world, cameras, video_writer, recording, on_screen_render):
    global shared_world, shared_cameras, shared_video_writer, shared_recording, shared_on_screen_render
    shared_world = world
    shared_cameras = cameras
    shared_video_writer = video_writer
    shared_recording = recording
    shared_on_screen_render = on_screen_render
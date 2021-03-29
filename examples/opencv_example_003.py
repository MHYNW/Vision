## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# 640 x 480, 30fps
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
''' RGB Image
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
'''
# average pooling 
def average_pooling(img, G=8):
    out = img.copy()
    H, W  = img.shape
    Nh = int(H / G)
    Nw = int(W / G)
    for y in range(Nh):
        for x in range(Nw):
             out[G*y:G*(y+1), G*x:G*(x+1)] = np.max(out[G*y:G*(y+1), G*x:G*(x+1)]).astype(np.int)
    return out

# Start streaming
profile = pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            print("emitter yeeeees")
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        if depth_sensor.supports(rs.option.laser_power):
            print("laser yesssss")
            range = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, range.max)

        # visual preset number can be change, didn't fixed yet. can be 0. and maybe it means the high accuracy of the sensor
        # depth_sensor.set_option(rs.option.visual_preset, 0)
        depth_scale = depth_sensor.get_depth_scale()
       # color_frame = frames.get_color_frame()
        '''
        if not depth_frame or not color_frame:
            continue
        '''
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        scale_image = np.asanyarray(depth_scale)
        # pool_image = average_pooling(depth_image)
        distance_image = cv2.convertScaleAbs(depth_image, alpha=1, beta=0) 
        # color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)

        # Canny edge detector
        edge = cv2.Canny(depth_colormap, 50, 150)
        # pool_image = average_pooling(edge)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        # Center cross
        cv2.line(distance_image, (310, 240), (330, 240), (0, 0, 200), 2)
        cv2.line(distance_image, (320, 230), (320, 250), (0, 0, 200), 2)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', distance_image)
        # cv2.imshow('RealSense', scale_image)
        cv2.waitKey(1)
        # print("depth: {0}, distance: {1}".format(depth_image(320, 240), depth_frame.get_distance(320,240)))
        print("distnace: {}".format(depth_frame.get_distance(320, 240)))
        print("depth: {}".format(depth_image[320,240]))
        print("distnce estimate: {}".format(depth_scale*depth_image[320,240]))
        print("depth scale: {}".format(depth_scale))

finally:
    # Stop streaming
    pipeline.stop()

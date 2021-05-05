## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
## 2020.05.~ ver. dependant with size of UAV
###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import velocity_generator
import sys
import math 
import threading

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# Depth Image, 640 x 480, 30fps
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# RGB Image, 960 x 540, 30fps
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# distance importing
vel = velocity_generator.vel_generator()
vel.RandAccGen()


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

# Set Depth sensor and scale
depth_sensor = profile.get_device().first_depth_sensor()

if depth_sensor.supports(rs.option.emitter_enabled):
    print("emitter on")
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

if depth_sensor.supports(rs.option.laser_power):
    print("laser on")
    range = depth_sensor.get_option_range(rs.option.laser_power)
    depth_sensor.set_option(rs.option.laser_power, range.max)

depth_scale = depth_sensor.get_depth_scale()

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # velocity importing
        velocity = vel.vel
        vector_direction = vel.vel_dir
        acc = 1 # [m/s^2]
        print("velocity direction: {}".format(vector_direction))
        if vector_direction == 1:
            clipping_distance = velocity[0, 0]**2/2*acc
        else:
            clipping_distance = 1
        
        clipping_depth = clipping_distance/depth_scale


        # std
        '''
        std_depth_image = np.std(depth_image)
        mean_depth_image = np.mean(depth_image)
        clipping_depth = std_depth_image + mean_depth_image
        clipping_distance = depth_scale * clipping_depth
        print("std: {}".format(std_depth_image))
        print("mean: {}".format(mean_depth_image))
        print("clipping distance : {}".format(clipping_distance))
        '''

        # Set background
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_depth) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', bg_removed)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


        '''
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        '''

        '''
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            print("emitter onnnnnnnn")
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        if depth_sensor.supports(rs.option.laser_power):
            print("laser oooooooon")
            range = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, range.max)
        '''

        # visual preset number can be change, didn't fixed yet. can be 0. and maybe it means the high accuracy of the sensor
        # depth_sensor.set_option(rs.option.visual_preset, 0)
        # depth_scale = depth_sensor.get_depth_scale()
        '''
        if not depth_frame or not color_frame:
            continue
        if not depth_frame:
            continue
        '''

        ''' TBD

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        scale_image = np.asanyarray(depth_scale)
        # pool_image = average_pooling(depth_image)
        distance_image = cv2.convertScaleAbs(depth_image, alpha=1, beta=0) 
        # color_image = np.asanyarray(color_frame.get_data())
        '''

        '''
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)
        '''
        ''' edge detector(TBD)
        # Canny edge detector
        edge = cv2.Canny(depth_colormap, 50, 150)
        '''
        # pool_image = average_pooling(edge)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        '''
        # Center cross
        cv2.line(distance_image, (310, 240), (330, 240), (0, 0, 200), 2)
        cv2.line(distance_image, (320, 230), (320, 250), (0, 0, 200), 2)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', distance_image)
        # cv2.imshow('RealSense', scale_image)
        cv2.waitKey(1)
        print("distnace: {}".format(depth_frame.get_distance(320, 240)))
        print("depth: {}".format(depth_image[320,240]))
        print("distnce estimate: {}".format(depth_scale*depth_image[320,240]))
        print("depth scale: {}".format(depth_scale))
        '''

finally:
    # Stop streaming
    pipeline.stop()

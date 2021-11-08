## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import sys
sys.path.append('/usr/local/lib/python3.6/pyrealsense2')

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
from torchvision import transforms
from collections import Counter

def find_device_that_supports_advanced_mode() :
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07",
                       "0B3A", "0B5C"]
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No D400 product line device that supports advanced mode was found")

def set_advanced_mode_from_json(jsonfilepath):
    try:
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        # Loop until we successfully enable advanced mode
        while not advnc_mode.is_enabled():
            print("Trying to enable advanced mode...")
            advnc_mode.toggle_advanced_mode(True)
            # At this point the device will disconnect and re-connect.
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            # The 'dev' object will become invalid and we need to initialize it again
            dev = find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        with open(jsonfilepath,"r") as f:
            as_json_object = json.load(f)
            if type(next(iter(as_json_object))) != str:
                as_json_object = {k.encode('utf-8'): v.encode("utf-8") for k, v in as_json_object.items()}
            json_string = str(as_json_object).replace("'", '\"')
            advnc_mode.load_json(json_string)
            print('setting over')
    except Exception as e:
        print(e)
        pass

def filters_config(decimation_scale):

    decimation = rs.decimation_filter()

    # you can also increase the following parameter to decimate depth more (reducing quality)
    decimation.set_option(rs.option.filter_magnitude, decimation_scale)

    spatial = rs.spatial_filter()
    # spatial.set_option(rs.option.filter_magnitude, 5)
    # spatial.set_option(rs.option.filter_smooth_alpha, 1)
    # spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)  # 5 = fill all the zero pixels

    temporal = rs.temporal_filter()

    hole_filling = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    return decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth

def depth_processing(depth_frame, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth):

    # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # print('depth frame', np.shape(colorized_depth))
    #cv2.namedWindow('Depth Frame', cv2.WINDOW_NORMAL)
    #cv2.imshow('Depth Frame', colorized_depth)

    # If you are satisfied with lower spatial resolution, the Decimation Filter will reduce spatial resolution
    # preserving z-accuracy and performing some rudimentary hole-filling.
    # decimated_depth = decimation.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    # print('decimated depth', np.shape(colorized_depth))
    #cv2.imshow('Decimated Depth', colorized_depth)

    disparity_depth = depth_to_disparity.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(disparity_depth).get_data())
    # print('disparity depth', np.shape(colorized_depth))
    #cv2.imshow('Disparity Depth', colorized_depth)

    # Spatial Filter is a fast implementation of Domain-Transform Edge Preserving Smoothing
    filtered_depth = spatial.process(disparity_depth)
    # colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    # print('spatial filtered depth', np.shape(colorized_depth))
    #cv2.imshow('Spatial Filtered Depth', colorized_depth)

    # Our implementation of Temporal Filter does basic temporal smoothing and hole-filling.
    temp_filtered = temporal.process(filtered_depth)
    # colorized_depth = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
    # print('temporal filtered depth', np.shape(colorized_depth))
    #cv2.imshow('Temp Filtered Depth', colorized_depth)

    to_depth = disparity_to_depth.process(temp_filtered)

    filled_depth = hole_filling.process(to_depth)
    # colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    # print('filled depth', np.shape(colorized_depth))
    # print('original filled depth', np.shape(np.asanyarray(filled_depth.get_data())))
    #cv2.imshow('Hole Filled Depth', colorized_depth)

    return filled_depth
    
def initial_camera():


    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    
    jsonfilepath = './form2fit/code/ml/dataloader/HighDensityPreset.json'
    set_advanced_mode_from_json(jsonfilepath)


    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline,align

def get_curr_image(pipeline,align):

    decimation_scale = 2
    wait_frame_count = 30
    decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth = filters_config(decimation_scale)
    # colorizer = rs.colorizer(color_scheme=3)
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 424x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        #guaratee that depth frames are useable
        aligned_depth_data = np.asanyarray(aligned_depth_frame.get_data()).astype('uint8')
        if Counter(aligned_depth_data.ravel())[0] > 0.2 * 848 * 480: 
            continue

        #再等待30帧，图片亮度会有提升
        if wait_frame_count > 0:
            wait_frame_count = wait_frame_count - 1 
            continue
        
        processed_frame = depth_processing(aligned_depth_frame, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth)
        processed_depth = processed_frame.as_depth_frame()
        depth_image = np.asanyarray(processed_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        depth_image = depth_image.astype('uint8')

        # if depth_image.ndim == 3:
        #     if depth_image.shape[2]==3:
        #         depth_image = cv2.cvtColor(depth_image,cv2.COLOR_BGR2GRAY)
        #     #depth_image = np.squeeze(depth_image,axis=2)

        if color_image.ndim == 3:
            if color_image.shape[2]==3:
                color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        #     #color_image = np.squeeze(color_image,axis=2)

        # assert depth_image.shape == (480,848)
        # assert color_image.shape == (480,848)
        # H,W = depth_image.shape 
        # cv2.imshow('depth',depth_image)
        # cv2.imshow('depth',color_image)
        print('color.shape',color_image.shape)
        print('depth.shape',depth_image.shape)
        cv2.imwrite('depth_image.png',depth_image)
        cv2.imwrite('color_image.png',color_image)
        # cv2.waitKey()
        break

    return color_image,depth_image

def norm_img(color_image,depth_image):

    #compute the mean and std of color and depth img
    color_mean = color_image.mean()
    color_std = color_image.std()
    depth_mean = depth_image.mean()
    depth_std = depth_image.std()

    _transform = transforms.ToTensor()
    _c_norm = transforms.Normalize(mean=color_mean, std=color_std)
    _d_norm = transforms.Normalize(mean=depth_mean, std=depth_std)

    color_tensor = _transform(color_image)
    color_norm = _c_norm(color_tensor)
    depth_tensor = _transform(depth_image)
    depth_norm = _d_norm(depth_tensor)

    return color_norm,depth_norm

if __name__ == '__main__':
    pipeline,align = initial_camera()
    get_curr_image(pipeline, align)
    print('1:',time.time())
    #img2 = get_curr_image(pipeline,align,clipping_distance)
    #print('1:', time.time())
    #img3 = get_curr_image(pipeline,align,clipping_distance)
    #print('1:', time.time())
    #print(img1.shape,img2.shape,img3.shape)
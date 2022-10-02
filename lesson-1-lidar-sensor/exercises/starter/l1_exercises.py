# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2

def load_range_image(frame, lidar_name):
    
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    return ri

# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")
    # extract range image from frame
    ri = load_range_image(frame, lidar_name)
    ri[ri<0]=0.0

    # map value range to 8bit
    # hinner dimensions of a range image cell are [range, intensity, elongation, is_in_no_label_zone]
    ri_intensity = ri[:,:,1]
    # ri_intensity = ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    # boost image
    ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    img = ri_intensity.astype(np.uint8)

    # focus on +/- 45° around the image center
    deg45 = int(img.shape[1] / 8)
    ri_center = int(img.shape[1]/2)
    img = img[:,ri_center-deg45:ri_center+deg45]

    print('max. val = ' + str(round(np.amax(img[:,:]),2)))
    print('min. val = ' + str(round(np.amin(img[:,:]),2)))

    cv2.imshow('intensity_image', img)
    cv2.waitKey(0)


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")
    # load range image
    ri = load_range_image(frame, lidar_name)
        
    # compute vertical field-of-view from lidar calibration 
    calibration = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    min = calibration.beam_inclination_min
    max = calibration.beam_inclination_max

    v_fov = max - min

    print(v_fov)
    print(ri.shape[0])

    # compute pitch resolution and convert it to angular minutes
    pitch_res_rad = v_fov / ri.shape[0]
    pitch_res_deg = pitch_res_rad * 180 / np.pi

    pitch_res_minutes = pitch_res_deg * 60

    print(pitch_res_deg)
    print(pitch_res_minutes)


    # solution in the course
    # # load range image
    # lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    # if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
    #     ri = dataset_pb2.MatrixFloat()
    #     ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
    #     ri = np.array(ri.data).reshape(ri.shape.dims)

    # # compute vertical field-of-view from lidar calibration 
    # lidar_calib = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0] # get laser calibration
    # min_pitch = lidar_calib.beam_inclination_min
    # max_pitch = lidar_calib.beam_inclination_max
    # vfov = max_pitch - min_pitch

    # # compute pitch resolution and convert it to angular degrees
    # pitch_res_rad = vfov / ri.shape[0]
    # pitch_res_deg = pitch_res_rad * 180 / np.pi
    # print("pitch angle resolution = " + '{0:.2f}'.format(pitch_res_deg) + "°")
    

# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    print("Exercise C1-3-1")    

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = 0
    
    # print(frame.laser_labels)

    for ll in frame.laser_labels:
        if ll.type == ll.TYPE_VEHICLE:
            num_vehicles += 1

    print("number of labeled vehicles in current frame = " + str(num_vehicles))
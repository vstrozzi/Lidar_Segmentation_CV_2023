import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from LidarMapping.depth_map import dense_map
import matplotlib.pyplot as plt
from LidarMapping.calibration import Calibration

def create_lidar_imposition(img_path, lidar_path, calib_path, level = 4):
    # Loading the image, LiDAR data and  Calibration
    img = cv2.imread(img_path)
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    calib = Calibration(calib_path)
    # From LiDAR coordinate system to Camera Coordinate system
    lidar_rect = calib.lidar2cam(lidar[:,0:3])
    # From Camera Coordinate system to Image frame
    lidarOnImage, mask = calib.rect2Img(lidar_rect, img.shape[1], img.shape[0])

    # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
    lidarOnImage2 = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
    #print(lidarOnImage2.T.shape)
    return dense_map(lidarOnImage2.T, img.shape[1], img.shape[0], level), lidarOnImage2
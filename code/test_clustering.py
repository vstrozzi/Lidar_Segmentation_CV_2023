#from sklearn.cluster import DBSCAN
#from LidarMapping.calibration import Calibration
import os
import cv2
import numpy as np
from LidarMapping.calibration import Calibration
from LidarMapping.depth_map import dense_map
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = "LidarMapping/test_data/"
    image_dir = os.path.join(root, "image_2")
    velodyne_dir = os.path.join(root, "velodyne")
    calib_dir = os.path.join(root, "calib")
    cur_id = 1
    # Loading the image, LiDAR data and  Calibration
    img = cv2.imread(os.path.join(image_dir, "%06d.png" % cur_id))
    lidar = np.fromfile(os.path.join(velodyne_dir, "%06d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
    calib = Calibration(os.path.join(calib_dir, "%06d.txt" % cur_id))
    lidar_rect = calib.lidar2cam(lidar[:,0:3])
    lidarOnImage, mask = calib.rect2Img(lidar_rect, img.shape[1], img.shape[0])
    lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
    out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 6)
    rows, cols, chs = out.shape
    indices = np.dstack(np.indices(out.shape[:2]))
    xycolors = np.concatenate((img, indices), axis=-1) 
    xycolors = np.reshape(xycolors, [-1,5])
    print(xycolors.shape)
    db = DBSCAN(eps=5, min_samples=50, metric = 'euclidean',algorithm ='auto')
    db.fit(xycolors)
    labels = db.labels_
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(np.reshape(labels, [rows, cols]))
    plt.axis('off')
    plt.show()

#dbscan = DBSCAN(eps = 8, min_samples = 4).fit(x) # fitting the model
#labels = dbscan.labels_ # getting the labels
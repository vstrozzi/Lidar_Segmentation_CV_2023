import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from depth_map import dense_map
import matplotlib.pyplot as plt
from calibration import Calibration

if __name__ == "__main__":
    root = "test_data/"
    image_dir = os.path.join(root, "image_2")
    velodyne_dir = os.path.join(root, "velodyne")
    calib_dir = os.path.join(root, "calib")
    cur_id = 1
    # Loading the image, LiDAR data and  Calibration
    img = cv2.imread(os.path.join(image_dir, "%06d.png" % cur_id))
    lidar = np.fromfile(os.path.join(velodyne_dir, "%06d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
    calib = Calibration(os.path.join(calib_dir, "%06d.txt" % cur_id))
    # From LiDAR coordinate system to Camera Coordinate system
    lidar_rect = calib.lidar2cam(lidar[:,0:3])
    # From Camera Coordinate system to Image frame
    lidarOnImage, mask = calib.rect2Img(lidar_rect, img.shape[1], img.shape[0])

    # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
    lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
    out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 6)
    plt.figure(figsize=(20,40))
    plt.imsave("depth_map_%06d.png" % cur_id, out)
    plt.close()
    
    exit()

    fig = plt.figure(figsize=(20,40))
    
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    #ax.set_facecolor("white")
    #plt.scatter(x_points, y_points, c=np.arange(len(x_points)), cmap='gist_rainbow')
    plt.imshow(out, alpha=0.5, cmap='gist_rainbow')
    plt.savefig("overlay_depth_map_%06d.png" % cur_id)
    plt.close()
    #plt.show()
 
    lidar_to_frame = calib.Img2frame(lidarOnImage.T, img.shape[1], img.shape[0])
    ## Visualize the concatenated image
    plt.figure(figsize=(10, 10))
    plt.imshow(lidar_to_frame, cmap='binary', interpolation='nearest' )
    plt.colorbar()
    plt.show()

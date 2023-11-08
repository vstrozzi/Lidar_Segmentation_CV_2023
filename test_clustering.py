#from sklearn.cluster import DBSCAN
#from LidarMapping.calibration import Calibration

import LidarMapping
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

#dbscan = DBSCAN(eps = 8, min_samples = 4).fit(x) # fitting the model
#labels = dbscan.labels_ # getting the labels
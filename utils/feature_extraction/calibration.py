import numpy as np

class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P = calibs['P2'] #Camera coordinates, rectified. Intrinsic metrics K
        self.P = np.reshape(self.P, [3,4])

        self.L2C = calibs['Tr_velo_to_cam'] # maps a point in point cloud coordinate to reference co-ordinate. That's my T
        self.L2C = np.reshape(self.L2C, [3,4])

        self.R0 = calibs['R0_rect'] # Rectified rotation
        self.R0 = np.reshape(self.R0,[3,3])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))
        pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(self.L2C))
        pts_3d_cam_rec = np.transpose(np.dot(self.R0, np.transpose(pts_3d_cam_ref)))
        return pts_3d_cam_rec
    
    def rect2Img(self, rect_pts, img_width, img_height):
        n = rect_pts.shape[0]
        points_hom = np.hstack((rect_pts, np.ones((n,1))))
        points_2d = np.dot(points_hom, np.transpose(self.P)) # nx3
        points_2d[:,0] /= points_2d[:,2]
        points_2d[:,1] /= points_2d[:,2]
        
        mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
        mask = mask & (rect_pts[:,2] > 2)
        return points_2d[mask,0:2], mask
    
    def Img2frame(self, Pts, n, m):
        mX = np.zeros((m,n)) + np.float("inf")
        mY = np.zeros((m,n)) + np.float("inf")
        mD = np.zeros((m,n))
        mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
        mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
        mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]

        rgb_image = np.stack((mX, mY, mD), axis=2)
        return rgb_image
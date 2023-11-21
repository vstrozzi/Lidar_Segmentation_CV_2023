import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from LidarMapping.calibration import Calibration
from create_lidar_imposition import create_lidar_imposition
DATASET_LENGTH = 2124
BASE = "img_"




class Lidarset(Dataset):
    CALIB_PATH = "calibrations"
    CALIB_NAME = "calib"
    IMAGE_PATH = "imgs"
    LIDAR_PATH  = "lidar"
    LIDAR_NAME  = "velodyne"

    DEPTH_NAME = "depth"
    DEPTH_FOLDER = "depth_map"
    MAPPED_LIDAR_NAME = "m_lidar"
    MAPPED_LIDAR_FOLDER = "mapped_lidar"

    def __init__(self, path, level = 6, verbose = True, save = False) -> None:
        super().__init__()
        self.verbose = verbose
        if(verbose) : print("Loading storages")
        
        self._path = path
        self._all_paths = os.listdir(self._path)
        #self._images_paths = [path for path in self._all_paths if self.IMAGE_PATH in os.path.basename(path)]
        self._imgs_path = os.path.join(self._path, self.IMAGE_PATH)
        self._calib_path = os.path.join(self._path, self.CALIB_PATH)
        self._lidar_path = os.path.join(self._path, self.LIDAR_PATH)
        #print(self._imgs_path, self._calib_path, self._lidar_path)
        self.level = level
        self._depth_map_folder = self._path+"/"+self.DEPTH_FOLDER
        self._mapped_lidar_folder = self._path+"/"+self.MAPPED_LIDAR_FOLDER
        

        #self.lidar_storage = cv2.FileStorage(lidar_path, cv2.FILE_STORAGE_READ)
        #if(verbose) : print("Loading Reflectance....")
        #self.reflectance_storage = cv2.FileStorage(reflectance_path, cv2.FILE_STORAGE_READ)
        #self.image_dir = camera_path
        #if(verbose) : print("Loading Camera....")
        #self.images = os.listdir(camera_path)
        #self.images.sort()
        #self.length = len(self.images)
        self.load(save)
        if(verbose) : print("Loading Complete")

    def __load_storage__(self, parent):
            temp_storage = {}
            for f in os.listdir(parent):
                temp_storage[f] = os.listdir(os.path.join(parent,f))
            return temp_storage
         
    def load(self, with_save = False):
        if(self.verbose) : print("Loading Images Names....")
        self._imgs_folder_storage = self.__load_storage__(self._imgs_path)
        #print(self._imgs_folder_storage)
        self._imgs_folder_list = self.__to_list__(self._imgs_folder_storage)
        if(self.verbose) : print("Loading Lidar Names....")
        self._lidar_folder_storage = self.__load_storage__(self._lidar_path)
        self._lidar_folder_list = self.__to_list__(self._lidar_folder_storage)
        #print(self._lidar_folder_storage)

        if(self.verbose) : print("Loading Calib Names....")
        self._calibrations_folder_storage = self.__load_storage__(self._calib_path)
        self._calibrations_folder_list= self.__to_list__(self._calibrations_folder_storage)
        #print(self._calibrations_folder_storage)
        

        # load per image 
        if with_save:
            if(self.verbose) : print("Saving Depth and Mapped Lidar Information....")
            #save path
            if self.DEPTH_FOLDER not in os.listdir(self._path):
                 #print("Nope")
                os.mkdir(self._depth_map_folder)
            
            if self.MAPPED_LIDAR_FOLDER not in os.listdir(self._path):
                 #print("Nope")
                os.mkdir(self._mapped_lidar_folder)
            

            for current_path in os.listdir(self._imgs_path):
                #print(current_path)
                current_imgs_path  =os.path.join(self._imgs_path,current_path)
                imgs = os.listdir(current_imgs_path)
                if len(imgs) == 0:
                    continue
                folder_number = current_path.split("_")[1]
                current_lidar_path = os.path.join(self._lidar_path,self.LIDAR_NAME +"_" + folder_number)
                current_calib_path = os.path.join(self._calib_path,self.CALIB_NAME +"_" + folder_number)
                current_depth_map_folder  =  os.path.join(self._depth_map_folder,self.DEPTH_NAME +"_" + folder_number)
                current_mapped_lidar_folder  =  os.path.join(self._mapped_lidar_folder,self.MAPPED_LIDAR_NAME +"_" + folder_number)
                if self.DEPTH_NAME +"_" + folder_number not in os.listdir(self._depth_map_folder):
                    #print("Nope")
                    os.mkdir(current_depth_map_folder)
                
                if self.MAPPED_LIDAR_NAME +"_" + folder_number not in os.listdir(self._mapped_lidar_folder):
                    #print("Nope")
                    os.mkdir(current_mapped_lidar_folder)

                for img_name in imgs:
                    #print(current_imgs_path, img_name)
                    obj_number = img_name.split(".")[0]

                    img_to_load = os.path.join(current_imgs_path, img_name)
                    lidar_to_load = os.path.join(current_lidar_path, obj_number + ".bin" )
                    calib_to_load = os.path.join(current_calib_path, obj_number + ".txt" )

                    out, mapped_lidar = create_lidar_imposition(img_to_load, lidar_to_load, calib_to_load, self.level)
                    #print(mapped_lidar.shape)
                    np.save(current_mapped_lidar_folder+f"/ml{obj_number}", mapped_lidar)
                    #plt.figure(figsize=(20,40))
                    #print(mapped_lidar)
                    #np.save('data/temp/np_save', a)
                    plt.imsave(current_depth_map_folder+f"/dp{obj_number}.png" , out)
                    #plt.close()
                    #print(img_to_load)
                    #print(lidar_to_load)
                    #print( calib_to_load)

        #print(self._depth_map_folder)
        if(self.verbose) : print("Loading DepthMap Names....")
        self._depth_folder_storage = self.__load_storage__(self._depth_map_folder)
        self._depth_folder_list = self.__to_list__(self._depth_folder_storage)

        if(self.verbose) : print("Loading Mapped Lidar Names....")
        self._mapped_lidar_folder_storage = self.__load_storage__(self._mapped_lidar_folder)
        self._mapped_lidar_folder_list = self.__to_list__(self._mapped_lidar_folder_storage)

        #print(self._depth_folder_storage)




        #img = 

    def __len__(self):
            return self.length
    def __to_list__(self, storage):
        result_list = []
        for key, values in storage.items():
            for value in values:
                result_list.append(f'{key}-{value}')

        return result_list
    def __getitem__(self, index):
        img_folder, img = self._imgs_folder_list[index].split("-")
        #lidar_folder, lidar = self._lidar_folder_list[index].split("-")
        #calib_folder,calib = self._calibrations_folder_list[index].split("-")
        depth_folder,depth = self._depth_folder_list[index].split("-")
        mlidar_folder,mlidar = self._mapped_lidar_folder_list[index].split("-")
        #print(img, lidar, calib, depth, mlidar)

        img_path = os.path.join(self._imgs_path, img_folder, img)
        #lidar_path = os.path.join(self._lidar_path, lidar_folder, lidar)
        #calib_path = os.path.join(self._calib_path, calib_folder, calib)
        mlidar_path = os.path.join(self._mapped_lidar_folder, mlidar_folder, mlidar)
        depth_path = os.path.join(self._depth_map_folder, depth_folder, depth)
        
        
        rgb_image = np.array(cv2.imread(img_path))
        #(1)LiDAR position with the intesity (3), with (2) we would have the depth
        mlidar_item = np.load(mlidar_path)
        rgb_depth = np.array(cv2.imread(depth_path))
        return rgb_image, mlidar_item, rgb_depth
            #lidar_image = self.lidar_storage.getNode(f'img_{index}').mat()
            #reflectance_image = self.reflectance_storage.getNode(f'img_{index}').mat()
            #
            #img_path = os.path.join(self.image_dir, self.images[index])
            #rgb_image = np.array(Image.open(img_path).convert("BGR"))
#
            #
            #return  lidar_image, reflectance_image,rgb_image




#img = cv2.imread(img_path)
 #   lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
  #  calib = Calibration(calib_path)

def test():
    dataset = Lidarset(path="./LidarMapping/test_data")
    rgb_image, mlidar_item, rgb_depth = dataset[1]
    
if __name__ == "__main__":
    test()
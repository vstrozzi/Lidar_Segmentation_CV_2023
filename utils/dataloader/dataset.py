"""
TorchDatset of KITTI 2015 Stereo dataset with stereo RGB pairs.

Dataset link: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
"""
import os
import random
import numpy as np
from easydict import EasyDict
from skimage import io
from PIL import Image
from torchvision import transforms  
from torch.utils.data.dataset import Dataset
from utils.dataloader.labels import *
import torch
from collections import defaultdict
from torchvision.transforms import InterpolationMode

class DatasetKITTI2015(Dataset):
    FIXED_SHAPE = (320, 1216)   # CROP
    REDUCED_SHAPE = (160, 808)

    def __init__(self, root_dir, mode, output_size, random_sampling=None, fix_random_seed=False):
        # Check arguments
        assert mode in ['training'], 'Invalid mode for DatasetKITTI2015'
        self.root_dir = root_dir
        self.mode = mode
        self.output_size = output_size


        if random_sampling is None:
            self.sampler = None
        elif isinstance(random_sampling, float):
            self.sampler = UniformSamplerByPercentage(random_sampling)
        else:
            raise ValueError

        if fix_random_seed:
            random.seed(420)
            np.random.seed(seed=420)

        # Get all data path
        self.left_data_path= get_kitti2015_datapath(self.root_dir, self.mode, self.sampler)

        # Define data transform
        self.transform = EasyDict()
        if self.mode in ['training']:
            self.transform.rgb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize(self.REDUCED_SHAPE, interpolation=InterpolationMode.BILINEAR, antialias=None)
            ])
            self.transform.depth = transforms.Compose([
                transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
                transforms.ToTensor(),
                transforms.Resize(self.REDUCED_SHAPE, interpolation=InterpolationMode.BILINEAR, antialias=None)
            ])
            self.transform.segm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.REDUCED_SHAPE, interpolation=InterpolationMode.NEAREST_EXACT, antialias=None)
            ])
        else: # val
            self.transform.rgb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
                transforms.Resize(self.REDUCED_SHAPE, interpolation=InterpolationMode.BILINEAR, antialias=None)
            ])
            self.transform.depth = transforms.Compose([
                transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
                transforms.ToTensor(),
                transforms.Resize(self.REDUCED_SHAPE, interpolation=InterpolationMode.BILINEAR, antialias=None)
            ])
            self.transform.segm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.REDUCED_SHAPE, interpolation=InterpolationMode.NEAREST_EXACT, antialias=None)
            ])

    def __getitem__(self, idx):
        # Get data
        
        left_rgb = read_rgb(self.left_data_path['rgb'][idx])

        img_h, img_w = left_rgb.shape[:2]
        left_disp = read_depth(self.left_data_path['disp'][idx])
        left_segm = read_segm(self.left_data_path['segm'][idx])

        if self.sampler is None:
            left_sdisp = depth2disp(read_depth(self.left_data_path['sdepth'][idx]))
        else:
            left_sdisp = self.sampler.sample(left_disp)

        # Crop to fixed size
        def crop_fn(x):
            start_h = img_h - self.FIXED_SHAPE[0] if img_h > self.FIXED_SHAPE[0] else 0
            start_w = 0
            end_w = min(img_w, start_w+self.FIXED_SHAPE[1])
            return x[start_h:start_h+self.FIXED_SHAPE[0], start_w:end_w]
        left_rgb, left_sdisp, left_disp, left_segm = list(map(crop_fn, [left_rgb, left_sdisp, left_disp, left_segm]))
        
        # Perform transforms
        data = dict()
        data['left_rgb']= self.transform.rgb(left_rgb)
        data['left_sdisp'] = self.transform.depth(left_sdisp)
        data['left_disp'] = self.transform.depth(left_disp)

        data['width'] = img_w

        return data, self.transform.segm(left_segm)

    def __len__(self):
        return len(self.left_data_path['rgb'])


class UniformSamplerByPercentage(object):
    """ (Numpy) Uniform sampling by number of sparse points """
    def __init__(self, percent_samples):
        super(UniformSamplerByPercentage, self).__init__()
        self.percent_samples = percent_samples
        self.max_depth = 100

    def sample(self, x):
        s_x = np.zeros_like(x)
        if self.max_depth is np.inf:
            prob = float(self.n_samples) / x.size
            mask_keep = np.random.uniform(0, 1, x.shape) < prob
            s_x[mask_keep] = x[mask_keep]
        else:
            sparse_mask = (x <= self.max_depth) & (x > 0)
            n_keep = sparse_mask.astype(float).sum()
            if n_keep == 0:
                raise ValueError('`max_depth` filter out all valid depth points')
            else:
                mask_keep = np.random.uniform(0, 1, x[sparse_mask].shape) < self.percent_samples
                tmp = np.zeros(mask_keep.shape)
                tmp[mask_keep] = x[sparse_mask][mask_keep]
                s_x[sparse_mask] = tmp

        return s_x


def depth2disp(depth):
    """ Convert depth to disparity for KITTI dataset.
        NOTE: depth must be the original rectified images.
        Ref:  """
    baseline = 0.54
    width_to_focal = dict()
    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.856
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
    width_to_focal[1238] = 718.3351

    focal_length = width_to_focal[depth.shape[1]]
    invalid_mask = depth <= 0
    disp = baseline * focal_length / (depth + 1E-8)
    disp[invalid_mask] = 0
    return disp


def read_segm(path):
    """ Read raw RGB SGB and perform MASK PROCESS to it, return HxWx1"""
    seg = io.imread(path)
    return RGBtoOneHot(seg, {x.color:((x.trainId)) for x in labels})

def read_rgb(path):
    """ Read raw RGB and DO NOT perform any process to the image """
    rgb = io.imread(path)
    return rgb


def read_depth(path):
    """ Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
    """
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32) / 256.0
    return depth[:, :, np.newaxis]
 

def get_kitti2015_datapath(root_dir, mode, sampler=None):
    """ Read path to all data from KITTI Stereo 2015 dataset """
    left_data_path = {'rgb': [], 'sdepth': [], 'disp': [], 'disp_occ': [], 'segm': []}
    if sampler is None:
        fname_list = sorted(os.listdir(os.path.join(root_dir, mode, 'velodyne_2')))
        fname_list = [f[:-4]+'_10.png' for f in fname_list]
    else:
        fname_list = sorted(os.listdir(os.path.join(root_dir, mode, 'image_2')))
        fname_list = [f for f in fname_list if f[-6:]=='10.png']
    for fname in fname_list:
        left_data_path['rgb'].append(os.path.join(root_dir, mode, 'image_2', fname))
        left_data_path['disp'].append(os.path.join(root_dir, mode, 'disp_noc_0', fname))
        left_data_path['disp_occ'].append(os.path.join(root_dir, mode, 'disp_occ_0', fname))
        left_data_path['segm'].append(os.path.join(root_dir, mode, 'labels_2', fname))
        if sampler is None:
            left_data_path['sdepth'].append(os.path.join(root_dir, mode, 'velodyne_2', fname[:-7]+'.png'))
    return left_data_path
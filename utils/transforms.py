import cv2
import torch
import numpy as np
from torchvision import transforms
from skimage.filters import threshold_multiotsu
from skimage import feature

transform_default = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

rgb_to_gray = transforms.Lambda(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
equalize_hist = transforms.Lambda(lambda image: cv2.equalizeHist(image).astype(np.uint8))

transform_mo = transforms.Compose([
    transforms.Lambda(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
    transforms.Lambda(lambda image: np.digitize(image, threshold_multiotsu(image, classes=2)).astype(np.uint8)),
])

# For canny you need ToPIL before and ToTensor after
transform_canny = transforms.Lambda(lambda image: np.array(feature.canny(np.array(image), sigma=2)).astype(np.uint8))

class lbl_for_patch(object):
    """Convert complete image label (tensor) to patchwise classification (tensor)"""

    def __init__(self, patch_size, img_shape, road_percentage=0.25):
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.road_percentage = road_percentage

    def __call__(self, lbl):
        num_patches = self.img_shape[0]//self.patch_size[0] * self.img_shape[1]//self.patch_size[1]
        out = torch.empty((num_patches, 1))
        lbl = lbl.squeeze()
        # Iter over patches left to right, start at top left
        c = 0
        for x in range(0, self.img_shape[0], self.patch_size[0]):
            for y in range(0, self.img_shape[1], self.patch_size[1]):
                # Over 25%
                out[c] = (1 if (lbl[x:x + self.patch_size[0], y:y + self.patch_size[1]].mean() >= self.road_percentage) else 0)
                c += 1
        return out
    
# This need both ToPIL and ToTensor before
transform_blur = transforms.GaussianBlur(3, 3)


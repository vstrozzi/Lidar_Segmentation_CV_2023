
from .tranformers.modules import *
from .basemodel import BaseModel

# Simple ViT class (using default parameters of vit_b_16 from paper An image is worth 16x16 ...)
# In: batches x FullImgs
# Output batches x num_patches_per_img ({0, 1} values)
class ViT(BaseModel):
    def __init__(self, loss, eval_metric, optimizer=torch.optim.Adam, lr=1e-4, image_size=(400, 400), patch_size=(16, 16), num_classes=2, dim=768, depth=12, heads=12,
                 mlp_dim=3072, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.2, emb_dropout = 0.):
        super().__init__(loss, eval_metric, optimizer, lr)
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        # Need images and patches to be divisor
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.embedder = Embedder(dim, image_size, patch_size, channels, emb_dropout = 0.)
        self.encoder = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.decoder = Decoder(dim, num_classes, self.num_patches)

    def forward(self, img):
        x = self.embedder(img)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
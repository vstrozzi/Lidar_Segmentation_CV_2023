from .blocks import *
class Embedder(nn.Module):
    def __init__(self, dim, image_size, patch_size, channels, emb_dropout = 0.):
        super().__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x
class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # Build encoder blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class Decoder(nn.Module):
    def __init__(self, dim, num_classes, num_patches):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.to_latent = nn.Identity()
        # Here The MLP_head outputs a value {0,1} for each patch
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):
        b, n, _ = x.shape
        # Here we have the segmentation logic
        # For each patch 16x16 (embedded with dimension = dim) we output a prediction 0,1 (ignore CLS for now)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        for j in range(b):
            for i in range(self.num_patches):
                # Extract patch embedding
                x_emb = x[j, i + 1,:]
                x_emb = self.to_latent(x_emb)
                x[j, i, 0] = torch.argmax(self.mlp_head(x_emb))
        x = x[:, 0:self.num_patches, 0:1]
        return x
    
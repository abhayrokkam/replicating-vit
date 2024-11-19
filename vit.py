import torch

class DataEmbeddings(torch.nn.Module):
    """
    
    """
    
    # Patches
    PATCH_SIZE = (16, 16)
    NUM_PATCHES = int((224 / 16) ** 2)

    # Patches to Embeddings
    EMBED_DIMS = 768
    
    def __init__(self,
                 in_channels = 3,
                 patch_size = PATCH_SIZE,
                 num_patches = NUM_PATCHES,
                 embed_dims = EMBED_DIMS) -> None:
        super().__init__()
        
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels,
                                          out_channels=embed_dims,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
        
        self.class_embedding = torch.nn.Parameter(torch.randn(size=(1, embed_dims)),
                                                  requires_grad=True)
        
        self.positional_embedding = torch.nn.Parameter(torch.randn(size=(num_patches + 1, embed_dims)),
                                                       requires_grad=True)

    def forward(self, x):
        # Extracting the shape
        b, c, h, w = x.shape
        
        # Patch Embeddings
        x = self.conv_layer(x)
        x = self.flatten(x)
        
        # Rearranging the dimensions
        x = x.permute(0, 2, 1)
        
        # Class Embedding
        x = torch.cat([self.class_embedding.expand(b, -1, -1), x], dim=1)
        
        # Positional Embeddings
        x += self.positional_embedding.expand(b, -1, -1)
        
        return x
        
        
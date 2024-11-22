import torch

from typing import Tuple

class DataEmbeddings(torch.nn.Module):
    """ 
    """
    def __init__(self,
                 in_channels: int,
                 patch_size: int | Tuple[int, int],
                 num_patches: int,
                 embed_dims: int) -> None:
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
        
class EncoderBlock(torch.nn.Module):
    """
    """
    def __init__(self,
                 embed_dims: int,
                 num_attn_heads: int,
                 ratio_hidden_mlp: int) -> None:
        super().__init__()
        
        self.layer_norm = torch.nn.LayerNorm(embed_dims)
        
        self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim=embed_dims,
                                                           num_heads=num_attn_heads,
                                                           batch_first=True)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=embed_dims,
                            out_features=int(embed_dims * ratio_hidden_mlp)),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=int(embed_dims * ratio_hidden_mlp),
                            out_features=embed_dims)
        )
        
    def forward(self, x):
        # Multi-head attention block
        x = self.layer_norm(x)
        attn_output, _ = self.multi_head_attn(query=x,
                                              key=x,
                                              value=x)
        x = attn_output + x
        
        # MLP block
        x = self.layer_norm(x)
        mlp_output = self.mlp(x)
        x = mlp_output + x
        
        return x
    
class ViT(torch.nn.Module):
    """
    """
    def __init__(self,
                 in_channels: int,
                 out_dims: int,
                 patch_size: int | Tuple[int, int],
                 num_patches: int,
                 embed_dims: int,
                 num_attn_heads: int,
                 ratio_hidden_mlp: int,
                 num_encoder_blocks: int) -> None:
        super().__init__()
        
        self.data_embeddings = DataEmbeddings(in_channels=in_channels,
                                              patch_size=patch_size,
                                              num_patches=num_patches,
                                              embed_dims=embed_dims)
        
        self.encoder_blocks = torch.nn.Sequential(*[
            EncoderBlock(embed_dims=embed_dims,
                         num_attn_heads=num_attn_heads,
                         ratio_hidden_mlp=ratio_hidden_mlp) for _ in range(num_encoder_blocks)
        ])
        
        self.classifier = torch.nn.Linear(in_features=embed_dims,
                                          out_features=out_dims)
        
    def forward(self, x):
        x = self.data_embeddings(x)
        x = self.encoder_blocks(x)
        x = self.classifier(x)
        
        return x
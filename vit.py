import torch

from typing import Tuple

class DataEmbeddings(torch.nn.Module):
    """ 
    A PyTorch module that generates patch embeddings for images using a convolutional layer, 
    along with learnable class and positional embeddings for transformer-based models.

    Args:
        in_channels (int): The number of input channels (e.g., 3 for RGB images).
        patch_size (int | Tuple[int, int]): The size of each patch (height, width).
        num_patches (int): The number of patches to extract from the image.
        embed_dims (int): The dimensionality of the embeddings for each patch.

    Attributes:
        conv_layer (torch.nn.Conv2d): Convolutional layer for extracting image patches.
        flatten (torch.nn.Flatten): Flattens the patch embeddings.
        class_embedding (torch.nn.Parameter): Learnable class embedding.
        positional_embedding (torch.nn.Parameter): Learnable positional embedding.

    Forward pass:
        - Extracts patches from the input image.
        - Adds class and positional embeddings to the patch embeddings.
        - Returns the final embedding tensor.

    Example:
        model = DataEmbeddings(in_channels=3, patch_size=16, num_patches=196, embed_dims=768)
        output = model(input_tensor)
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
    A transformer encoder block consisting of a multi-head attention layer followed by a feed-forward 
    MLP (multi-layer perceptron) block. Each block uses layer normalization and residual connections.

    Args:
        embed_dims (int): The dimensionality of the input and output embeddings.
        num_attn_heads (int): The number of attention heads in the multi-head attention layer.
        ratio_hidden_mlp (int): The ratio to scale the hidden layer size in the MLP (e.g., embed_dims * ratio).

    Attributes:
        layer_norm (torch.nn.LayerNorm): Layer normalization applied before attention and MLP blocks.
        multi_head_attn (torch.nn.MultiheadAttention): Multi-head attention mechanism for processing input.
        mlp (torch.nn.Sequential): A feed-forward MLP block with two linear layers and GELU activation.

    Forward pass:
        - Applies multi-head attention with residual connection.
        - Applies MLP with residual connection after layer normalization.
        - Returns the final output after both blocks.

    Example:
        encoder_block = EncoderBlock(embed_dims=768, num_attn_heads=12, ratio_hidden_mlp=4)
        output = encoder_block(input_tensor)
    """
    def __init__(self,
                 embed_dims: int,
                 num_attn_heads: int,
                 ratio_hidden_mlp: int,
                 batch_first: bool = True) -> None:
        super().__init__()
        
        self.layer_norm = torch.nn.LayerNorm(embed_dims)
        
        self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim=embed_dims,
                                                           num_heads=num_attn_heads,
                                                           batch_first=batch_first)
        
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
        x = x + attn_output
        
        # MLP block
        x = self.layer_norm(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        
        return x
    
class ViT(torch.nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    A transformer-based model that splits an input image into patches, embeds them, processes them 
    through multiple encoder blocks with attention mechanisms, and classifies the result.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        out_dims (int): Number of output classes.
        patch_size (int or Tuple[int, int]): Size of the image patches.
        num_patches (int): Total number of patches.
        embed_dims (int): Embedding dimension size.
        num_attn_heads (int): Number of attention heads in the encoder.
        ratio_hidden_mlp (int): Ratio of hidden layer size to embedding size.
        num_encoder_blocks (int): Number of transformer encoder blocks.

    Attributes:
        data_embeddings (DataEmbeddings): Embeds input image patches.
        encoder_blocks (torch.nn.Sequential): Transformer encoder blocks.
        classifier (torch.nn.Sequential): Classifier for output predictions.

    Forward pass:
        x (torch.Tensor): Input tensor (batch of images).
        Returns:
            torch.Tensor: Predicted class scores.
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
                         ratio_hidden_mlp=ratio_hidden_mlp) for _ in range(num_encoder_blocks)],
            torch.nn.LayerNorm(embed_dims))
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=embed_dims,
                            out_features=ratio_hidden_mlp*embed_dims),
            torch.nn.Linear(in_features=ratio_hidden_mlp*embed_dims,
                            out_features=out_dims)
        )
        
    def forward(self, x):
        x = self.data_embeddings(x)
        x = self.encoder_blocks(x)
        
        # Selecting the learnable embeddings (class embedding)
        x = x[:, 0, :]  
        
        x = self.classifier(x)
        
        return x
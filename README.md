# Replicating Vision Transformers (ViT)

## Introduction

The introduction of the transformer architecture by [**Vaswani et. al.**](https://doi.org/10.48550/arXiv.1706.) had the most significant impact on the field of artificial intelligence over the last decade. The architecture was so robust that only minimal changes have been proposed since 2017, with the most notable modification being the rearrangement of the layer normalization block. Having a pair of encoder-decoder while completely relying on attention to learn the patterns in the given sequence has proven to be effective enough to introduce the architecture for almost every use case in deep learning.

This project will focus on replicating the model proposed by [**Dosovitskiy et. al.**](https://doi.org/10.48550/arXiv.2010.11929), known as the Vision-Transformer (ViT). This paper introduces the transformer architecture (part of it) to the field of computer vision and image classification tasks. We will disect the parts of the ViT and implement it using PyTorch step-by-step.

**Note: All the hyperparameter numbers described in the examples below are directly extracted from the research paper. The numbers relate to the simplest ViT model.**

## Input to Encoder

We start by processing the image data which is fed to the transformer encoder. We will look at the encoder in a later part of this project. First, we will focus on converting our image data into the below shown format before feeding it to the transformer encoder.

**Overview** (just to read through, detailed explanation is below):
- Each image is chopped into multiple pieces (patches).
- Data that is in these patches is converted to patch embeddings.
- An extra learnable embedding is added (prepended) to the patch embeddings. This represents the class of our image (same as \<classification> token in BERT).
- Learnable embedding is the layer which will contain the useful information when we use the model for inference. The learnt information from the encoder will be stored here.
- Adding positional embeddings to give positional information regarding our data. 

<p align="center">
  <img src="./images/image_to_encoder.png"/>
  <br>
  <em>Looking at the input of the transformer encoder</em>
</p>

---

### Creating Patches

- Chop the image into `n` number of patches. This will be `num_patches`.

- The height and width of each patch will be `p`. This will be `patch_size`.

- Example:
    - An image of size `224 x 224` with `3` color channels will be of size `(3, 224, 224)`.

    - Each patch is of size `(16, 16)`. So, `patch_size = (16, 16)`. The number of chops along the height is `224 / 16 = 14`, and similarly, along the width, it is also `224 / 16 = 14`.

    - Having 14 pieces along height and 14 pieces along width makes the `num_patches = 14 * 14 = 196`.

    - We have `196` pieces of data for our encoder. Each patch of the image is treated sperately while feeding it to the encoder. 

---

### Patch Embeddings

- We have `num_patches` number of patches. Each patch is a piece of an image.

- These patches will have the size of a fixed size. This will be flattened and embedded into `d` number of dimensions. This will be `embed_dims`.

- Each patch is treated serperately. The dependencies and patterns across all the patches is learnt by the multi-head self attention layer in the encoder block (will be discussed later). 

- Example:
    - Each patch is of the size `16 x 16`.

    - This data will be projected (embedded) into `embed_dims` number of dimensions. Let us take `embed_dims = 768`.

    - Each patch will be turned to a linear vector which has the size of `768`. Total number of patches will be `196`.

    - The output will be of size `(768, 196)` which will be reshaped for better readability.

    - After reshaping it will be of size `(196, 768)` which shows that there are `196` patches of data with each patch having `768` dimensions.

#### Hybrid architecture

> As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection `E` (Eq. 1) is applied to patches extracted from a CNN feature map.

The above given text is from [**Dosovitskiy et. al.**](https://doi.org/10.48550/arXiv.2010.11929). It suggests that rather than flattening and embedding the patches, we can also apply an alternative approach. This is introducing CNN to extract feature maps from these images and then flattening these feature maps which will represent the linearly embedded data for each patch. This will be the approach for this project.

**CNN for embedding**:
- Using CNN with `kernel_size` of `16 x 16` which is the same as `patch_size`.

- Using `stride` value of `16 x 16` which is the same as `patch_size`.

- Using these two values will convert an image of size `(3, 224, 224)` into `(d, 14, 14)` as each patch will be converted to one single value from our CNN. Here, `d` is the `output_channels` of our CNN layer. This will be `embed_dims = 768` as it represents the number of dimensions that each patch has to be embedded into.

- The two dimensions of `14` will be flattened to have `196` patches of data. 

---

### Prepend Class Embedding

- After the computations of patch embeddings, a learnable dimension has to be prepended to the `patch_embeddings`.

- Learnable: The model will use this layer to learn the classes of our data. 

- Usually initialized with `randn` from the `torch` library. This will be a `nn.Parameter` object. 

- Using `nn.Parameter` will track gradients making the values modifiable through gradient descent which will be done during training.

- We will be using this layer as the input to our classifer block. (discussed in detail later)

- Example: 
    - The ouput of `patch_embeddings` will have the shape of `(196, 768)` for the number of patches and the dimension of each patch respectively.

    - A learnable classification embedding will be prepended to the `196` patches of data. The learnable layer will have the same dimensions as `embed_dims`. 

    - We will be using `torch.randn` to get random and normalized tensor which will be used in `nn.Paramter`.

    - We will be using `nn.Parameter` and setting `requires_grad = True` to make the values modifiable with gradient descent.

    - This will have the shape of `(1, 768)` which will be prepended to the patch embeddings.

    - The output after prepending will have the shape of `(197, 768)`

--- 

### Adding Positional Embeddings

- After the computations of all the required embeddings, the focus is to provide some information regarding the positions of the patches with respect to the whole image.

- This information regarding its positions will also be a learnable parameter. The model will learn the positional information during training. The explanation for this is given below.

- The process is same as the `class_embedding`.

- Example: 
    - We have the output from the `class_embedding` section which has the shape of `(197, 768)`.

    - Similar process of using `torch.randn` with `nn.Parameter` while setting `requires_grad = True`. This will be of shape `(197, 768)`

    - An element wise addition will be done using `positional_embeddings`. This addition provides the gateway to gradient descent.

    - Image shown below shows the plot of a learned positional embedding.

#### Why use positional embeddings?

<p align="center">
  <img src="./images/positional_embedding_plt.png"/>
  <br>
  <em>Learned positional embedding with `num_patches = 49`</em>
</p>

- This is to have some information regarding the postiions of our patches within the whole image. Imagining cutting an image into thousands of pieces and asking your friend to use these pieces to understand the whole image.

> The position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

- The paper does not use positional encodings from [**Vaswani et. al.**](https://doi.org/10.48550/arXiv.1706.) as the static values of the encoding process shows less improvements during the training for image data.

- The model sees better results with the positional embeddings where it is learnable and modifiable by the model during training.

- The problem arises while fine-tuning the model as the general consensus is to fine-tune a model on higher resolution data. Higher resolution images with the same `patch_size` will lead to higher number of patches. This makes the learnt positional embeddings during training to be useless. The solution is to perform interpolation of the pre-trained positional embeddings while respecting its position in images.

--- 
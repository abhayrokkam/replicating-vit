# Replicating Vision Transformers (ViT)

## 1. Introduction

The introduction of the transformer architecture by [**Vaswani et. al.**](https://doi.org/10.48550/arXiv.1706.) had the most significant impact on the field of artificial intelligence over the last decade. The architecture was so robust that only minimal changes have been proposed since 2017, with the most notable modification being the rearrangement of the layer normalization block. Having a pair of encoder-decoder while completely relying on attention to learn the patterns in the given sequence has proven to be effective enough to introduce the architecture for almost every use case in deep learning.

This project will focus on replicating the model proposed by [**Dosovitskiy et. al.**](https://doi.org/10.48550/arXiv.2010.11929), known as the Vision-Transformer (ViT). This paper introduces the transformer architecture (part of it) to the field of computer vision and image classification tasks. We will disect the parts of the ViT and implement it using PyTorch step-by-step.

**Note: All the hyperparameter numbers described in the examples below are directly extracted from the research paper. The numbers relate to the simplest ViT model.**

- [Replicating Vision Transformers (ViT)](#replicating-vision-transformers-vit)
  - [1. Introduction](#1-introduction)
  - [2. Input to Encoder](#2-input-to-encoder)
    - [2.1. Creating Patches](#21-creating-patches)
    - [2.2. Patch Embeddings](#22-patch-embeddings)
      - [2.2.1. Hybrid architecture](#221-hybrid-architecture)
    - [2.3. Prepend Class Embedding](#23-prepend-class-embedding)
    - [2.4. Adding Positional Embeddings](#24-adding-positional-embeddings)
      - [2.4.1. Why use positional embeddings?](#241-why-use-positional-embeddings)
      - [2.4.2. Fine-tuning Considerations](#242-fine-tuning-considerations)
  - [3. Transformer Encoder](#3-transformer-encoder)
    - [3.1. Layer Normalizations](#31-layer-normalizations)
    - [3.2. Residual Connections](#32-residual-connections)
    - [3.3. Multi-Head Self-Attention](#33-multi-head-self-attention)
    - [3.4. Multi-layer Perceptron](#34-multi-layer-perceptron)
  - [4. Classifier](#4-classifier)

## 2. Input to Encoder

We start by processing the image data which is fed to the transformer encoder. We will look at the encoder in a later part of this project. First, we will focus on converting our image data into the below shown format before feeding it to the transformer encoder.

**Overview**:

The input preparation involves several key steps:
1. Splitting the Image into Patches: Chopping the image into smaller pieces which will be called patches.
2. Generating Patch Embeddings: Converting each patch into a dense representation in a higher-dimensional space.
3. Adding a Learnable Class Embedding: Prepending a learnable vector that represents the image class. Same as `<classification>` token in BERT. Learnable embedding is the layer which will contain the useful information when we use the model for inference. The learnt information from the encoder will be stored here.
1. Incorporating Positional Embeddings: Adding positional information to patches to provide context about their arrangement within the image.

<p align="center">
  <img src="./images/input_to_encoder.png"/>
  <br>
  <em>Looking at the input of the transformer encoder</em>
</p>

---

### 2.1. Creating Patches

- The image is divided into `num_patches` pieces.

- The height and width of each patch is determined by the `patch_size`.

- Example:

For an image of size `(3, 224, 224)` where 3 is the number of color channels:

  - Patch size: `(16, 16)`.

  - Patches along height and width: `224 / 16 = 14`.

  - Total number of patches: `14 * 14 = 196`.

  - We have `196` pieces of data for our encoder. Each patch of the image is treated sperately while feeding it to the encoder. 

---

### 2.2. Patch Embeddings

- Each patch is flattened and projected into a vector of `embed_dims` dimensions using a linear layer.

- Each patch is treated serperately. The dependencies and patterns across all the patches is learnt by the multi-head self attention layer in the encoder block (will be discussed later).

- Example:

  - Each patch of size `(16, 16)` will be flattened to the size of `16 * 16 = 256`.

  - Each flattened patch will be embedded to `embed_dims = 768` number of dimensions.

  - With `196` patches, the output the patch embeddings section will be `(196, 768)`.

#### 2.2.1. Hybrid architecture

> As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection `E` (Eq. 1) is applied to patches extracted from a CNN feature map.

The above given text is from [**Dosovitskiy et. al.**](https://doi.org/10.48550/arXiv.2010.11929). It suggests that rather than flattening and embedding the patches, we can also apply an alternative approach. This is introducing CNN to extract feature maps from these images and then flattening these feature maps which will represent the linearly embedded data for each patch. This will be the approach for this project.

- **Implementation**:

  - Using CNN with `kernel_size` and `stride` of `(16, 16)` which is the same as `patch_size`.

  - Using these values with `embed_dims = 768` will convert an image of size `(3, 224, 224)` to `(768, 14, 14)` which is flattened into `(768, 196)`.

---

### 2.3. Prepend Class Embedding

- A learnable class embedding vector is prepended to the patch embeddings.

- Learnable: The model will use this layer to learn the classes of our data. This vector acts as a representative for the entire image and is updated during training.

- As it is learnable, it will be an object of `torch.nn.Parameter` making it modifiable during training.

- Example:

  - The ouput of `patch_embeddings` will have the shape of `(196, 768)` after reshaping.

  - Initialized as a random tensor using `torch.randn` and wrapped in `nn.Parameter` for gradient tracking.

  - The class embedding will have the shape of `(1, 768)` which will be prepended to the patch embeddings. The resulting shape will be `(197, 768)`.

--- 

### 2.4. Adding Positional Embeddings

To capture the spatial relationships among patches, learnable positional embeddings are added to the patch embeddings. This information regarding its positions will also be a learnable parameter. The model will learn the positional information during training.

- Example: 

  - A positional embedding tensor of shape `(197, 768)` is added element-wise to the patch embeddings.

  - Initialized similarly to class embeddings using `torch.randn` and `torch.nn.Parameter`.

#### 2.4.1. Why use positional embeddings?

- Positional embeddings provide context about the relative position of patches within the whole image. Imagine cutting an image into thousands of pieces and asking your friend to use these pieces to understand the whole image.
  
- Unlike fixed encodings (e.g., sinusoidal embeddings in [**Vaswani et al.**](https://doi.org/10.48550/arXiv.1706.)), learnable embeddings have been shown to yield better results for image data.

#### 2.4.2. Fine-tuning Considerations

- The problem arises while fine-tuning the model as the general consensus is to fine-tune a model on higher resolution data.

- Fine-tuning on higher-resolution images increases the number of patches, potentially invalidating pre-trained positional embeddings.

- A solution is to interpolate pre-trained positional embeddings to match the new patch count while preserving spatial relationships.
  
<p align="center">
  <img src="./images/positional_embedding_plt.png"/>
  <br>
  <em>Learned positional embedding with `num_patches = 49`</em>
</p>

## 3. Transformer Encoder

Once the input data is preprocessed, it is passed to the transformer encoder, the core component of the Vision Transformer (ViT) architecture. This section explains the structure and functionality of the transformer encoder, which learns relationships within the patches and stores the learned information in the class embedding layer.

**Overview**:

The transformer encoder block consists of:

1. Layer Normalization: Normalizes inputs to stabilize and improve training.
2. Multi-Head Self-Attention (MHSA): Extracts dependencies and relationships across patches.
3. Residual Connections: Ensures stable gradients and faster convergence by bypassing specific layers.
4. Multi-Layer Perceptron (MLP): Processes the attention outputs to higher dimensions before reverting to the original embedding size.
5. Repetition: Multiple encoder blocks can be stacked for enhanced learning.
6. Final Layer Normalization: Applied to the output of the last encoder block before feeding it into the classifier.

<p align="center">
  <img src="./images/transformer_encoder.png"/>
  <br>
  <em>Architecture of Transformer Encoder</em>
</p>

---

### 3.1. Layer Normalizations

- Purpose: Scaling the values of a single datapoint to stabalize training and reduce covariate shifts to make the model more capable of handling unseen and varied data.

- Example:
  
  - `normalized_shape`: Matches the embedding dimension (`embed_dims = 768`).

### 3.2. Residual Connections

- There are two residual connections in this architecture. These are the arrows that bypass different layers and lead to the `+` symbols. They take the values from the arrow-tail and add it to the values at the arrow-head (look at the graph).

- This helps to stabalize computations, speeds up convergence and reduces the impact of less significant features.
  
### 3.3. Multi-Head Self-Attention

The MHSA mechanism is central to the transformer encoder, enabling the model to learn dependencies between patches through self-attention.

- Process:

  - Inputs are transformed into three matrices: Query (`Q`), Key (`K`), and Value (`V`). These are the three arrows seen from normalization layer to MHSA layer.

  - An attention filter is computed using `Q` and `K`.

  - The filter is applied to `V`, resulting in a refined value matrix that emphasizes relevant features.

- Example:

  - `embed_dim`: Embedding dimension size (`768`).

  - `num_heads`: Number of attention heads (`12`). This is the 'multi-head' part of the layer. This will decide the number of self-attention layers to be used.

  - `batch_first`: Set to `True` as the first dimension of the input is the batch dimension.

  - All three `query`, `key`, `value` will take the same value which is the output from `LayerNorm`. This is because it is self-attention.

### 3.4. Multi-layer Perceptron

The MLP block processes the attention outputs by scaling them to a higher-dimensional space and then reducing them back to the original embedding dimensions.

- Purpose:

  - Captures complex patterns in the data.

  - Complements the attention mechanism by enabling further feature learning.

- Example:

  - First Layer:

    - `in_features` will be `embed_dims` (`768`).

    - `out_features` will be `mlp_ratio * embed_dims` (`4 * 768 = 3072`).

  - Second Layer:

    - `in_features` will be `mlp_ratio * embed_dims` (`3072`).

    - `out_features` will be `embed_dims` (`768`).

## 4. Classifier

The classifier is the final block in the Vision Transformer (ViT) pipeline, responsible for predicting the class of the input image. It processes the output from the Transformer Encoder to produce classification logits. The design is inspired by the original ViT paper, which outlines different configurations for pre-training and fine-tuning:

> *"The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time."* (Dosovitskiy et. al.)

1. Input to Classifier: The final output from the Transformer Encoder is normalized via a LayerNorm layer, resulting in a tensor of shape `(class_embedding + num_patches, embed_dims)`, typically (`197, 768`) for standard ViT configurations.

2. Classifier Layers:
   
    - Pre-training: An MLP with one hidden layer is used.

    - Fine-tuning: A single linear layer is preferrable.

3. MLP Details:

    - The input for the first linear layer is the class token (`class_embedding`), extracted as a tensor of shape `(1, embed_dims)`, which will be `(1, 768)`.

    - The first layer projects the embedding dimension (`768`) to a higher dimension based on the `mlp_ratio`, which will be `4 * 768`.

    - A hidden layer processes this expanded representation.

    - The second linear layer reduces the hidden dimensions back to `num_classes`, where `num_classes` represents the number of categories in the classification problem.

4. Output and Prediction:

    - The final output is a vector of logits for each class.

    - Apply a softmax function for a probability distribution across classes.

    - Use argmax to determine the class with the highest probability, representing the predicted category.

---
import torch
import torchmetrics

from pathlib import Path

from data_setup import get_dataloaders
from vit import ViT
from engine import train
from utils import create_writer, save_model

# Hyperparameters
COLOR_CHANNELS = 1
HEIGHT_WIDTH = (28, 28)

BATCH_SIZE = 32

PATCH_SIZE = (7, 7)
NUM_PATCHES = int((HEIGHT_WIDTH[0] / PATCH_SIZE[0]) ** 2)

EMBED_DIMS = 48
NUM_ATTN_HEADS = 6
RATIO_HIDDEN_MLP = 2
NUM_ENC_BLOCKS = 6

NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Donwload data if it hasn't been downloaded
data_path = Path('./data/')

# Get dataloaders
train_dataloader, test_dataloader, class_labels = get_dataloaders(batch_size=BATCH_SIZE)

model = ViT(in_channels=COLOR_CHANNELS,
            out_dims=len(class_labels),
            patch_size=PATCH_SIZE,
            num_patches=NUM_PATCHES,
            embed_dims=EMBED_DIMS,
            num_attn_heads=NUM_ATTN_HEADS,
            ratio_hidden_mlp=RATIO_HIDDEN_MLP,
            num_encoder_blocks=NUM_ENC_BLOCKS)

# Loss function, Accuracy
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
accuracy_function = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_labels))

    
# Writer (tensorboard)
writer = create_writer(model_name='vit',
                        experiment_name="model_" + "vit" + "_epochs_" + str(NUM_EPOCHS))

# Train
results = train(num_epochs=NUM_EPOCHS,
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_function=loss_function,
                optimizer=optimizer,
                accuracy_function=accuracy_function,
                device=device,
                writer=writer)

# Save the model
save_model(model=model,
            model_name="model_" + "vit" + "_epochs_" + str(NUM_EPOCHS) + ".pth",
            target_dir='./models/')

# Cuda memory management
del model
torch.cuda.empty_cache()

print("-"*100 + '\n')
"""
"""

import os
import random

import torch
import torchvision
import torchvision.datasets as datasets

import shutil
from pathlib import Path
from typing import List, Tuple

def data_download(target_classes: List[str] = ['cannoli', 'donuts', 'pancakes', 'tiramisu', 'waffles'],
                  amount_to_get: float = 0.25,
                  seed = 123):
    """
    Downloads and organizes a subset of the Food101 dataset into a target directory, 
    based on specified classes and a specified fraction of images per class.

    This function performs the following steps:
    1. Creates a local directory to store the data.
    2. Downloads the Food101 dataset (both training and testing splits).
    3. Selects a random subset of images from specified target classes.
    4. Copies the selected images into a structured target directory.
    5. Cleans up by removing the original downloaded dataset.

    Args:
        target_classes (List[str], optional): A list of class names to include from the Food101 dataset.
            Defaults to ['cannoli', 'donuts', 'pancakes', 'tiramisu', 'waffles'].
        amount_to_get (float, optional): The fraction of images to randomly select from each class.
            Defaults to 0.25, meaning 25% of the images in each class will be selected.
        seed (int, optional): The random seed for reproducibility of the image selection.
            Defaults to 123.

    Returns:
        None: This function does not return any values. It downloads and organizes the data 
        into directories on disk.

    Raises:
        FileNotFoundError: If the Food101 dataset is not available or cannot be downloaded.
        shutil.Error: If there is an issue copying images to the target directory.
    
    Example:
        data_download(target_classes=['cannoli', 'donuts'], amount_to_get=0.1)
            Downloads 10% of the images for 'cannoli' and 'donuts' from the Food101 dataset 
            and saves them in a structured directory under './data/desserts'.    
    """
    # Creating the data directory
    data_path = Path('./data/')
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)
    
    # Get training data
    train_data = datasets.Food101(root=data_path,
                                split="train",
                                download=True)

    # Get testing data
    test_data = datasets.Food101(root=data_path,
                                split="test",
                                download=True)
    
    # Setup data paths
    images_path = data_path / "food-101" / "images"

    ####################################################################

    # Function to separate a random amount of data
    def get_subset(images_path=images_path,
                data_splits=["train", "test"], 
                target_classes=target_classes,
                amount=amount_to_get,
                seed=seed):
        random.seed(seed)
        label_splits = {}
        
        # Get labels
        for data_split in data_splits:
            print(f"\n[INFO] Creating image split for: {data_split}...")
            label_path = data_path / "food-101" / "meta" / f"{data_split}.txt"
            
            class_images_path = []
            for target_class in target_classes:
                with open(label_path, "r") as f:
                    labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] == target_class]
                
                # Get random subset of target classes image ID's
                number_to_sample = round(amount * len(labels))
                print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split} in class {target_class}...")
                sampled_images = random.sample(labels, k=number_to_sample)
                
                # Apply full paths
                image_paths = [Path(str(images_path / sample_image) + ".jpg") for sample_image in sampled_images]
                class_images_path.extend(image_paths)
            
            label_splits[data_split] = class_images_path
                
        return label_splits
    
    ####################################################################
    
    # List of paths to randomly selected data
    label_splits = get_subset()
    
    # Create target directory path
    target_dir = Path("./data/desserts")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Moving the images to target dir
    for image_split in label_splits.keys():
        for image_path in label_splits[str(image_split)]:
            dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_dir)
    
    # Removing the downloaded data
    os.remove(Path(data_path / 'food-101.tar.gz'))
    shutil.rmtree(Path(data_path / 'food-101'))

def get_dataloaders(train_path: str,
                    test_path: str,
                    batch_size: int) -> Tuple[torch.utils.data.DataLoader, 
                                              torch.utils.data.DataLoader,
                                              List[str]]:
    """
    
    """
    # Dataset transforms
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size = (224, 224)),
        torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31),
        torchvision.transforms.ToTensor()]
    )

    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size = (224, 224)),
        torchvision.transforms.ToTensor()]
    )

    # Datasets
    train_dataset = torchvision.datasets.ImageFolder(root=train_path,
                                                    transform=train_transform,
                                                    target_transform=None)

    test_dataset = torchvision.datasets.ImageFolder(root=test_path,
                                                    transform=test_transform,
                                                    target_transform=None)
    
    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True)
    
    # Class labels
    class_labels = train_dataset.classes
    
    return train_dataloader, test_dataloader, class_labels
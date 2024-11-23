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
    Downloads and processes a subset of images from the Food101 dataset based on specified target classes.
    The images are split into training and testing sets, with a random subset of each class chosen based on 
    the amount_to_get parameter. The images are then moved into a target directory for further use.

    Args:
        target_classes (List[str]): A list of target class names (categories) to download. 
                                     Default is ['cannoli', 'donuts', 'pancakes', 'tiramisu', 'waffles'].
        amount_to_get (float): Fraction of the total available images per class to download, represented as 
                               a value between 0 and 1. Default is 0.25 (i.e., 25% of the images per class).
        seed (int): Random seed for reproducibility of random selections. Default is 123.

    Returns:
        None: This function modifies the filesystem by downloading, processing, and organizing image files 
              into appropriate directories. No value is returned.

    Raises:
        FileNotFoundError: If the Food101 dataset cannot be found or downloaded correctly.
        ValueError: If an invalid class name is provided in `target_classes`.
    
    Example:
        data_download(target_classes=['cannoli', 'donuts'], amount_to_get=0.5)
        This would download and move 50% of images from the 'cannoli' and 'donuts' classes from the Food101 dataset.

    Notes:
        - The function assumes that the Food101 dataset is not already present in the './data/' directory.
        - The images are saved into a subdirectory `./data/desserts/` organized by 'train' and 'test' splits.
        - Any existing data in `./data/desserts/` will be overwritten.
        - The function automatically cleans up the temporary files by removing the downloaded dataset archive and folder.
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

    # Setting random seed
    random.seed(seed)
    
    data_splits = ['train', 'test']
    label_splits = {}
    
    # Get labels
    for data_split in data_splits:
        print(f"\n[INFO] Creating image split for: {data_split}...")
        label_path = data_path / "food-101" / "meta" / f"{data_split}.txt"
        
        class_images_path = []
        for target_class in target_classes:
            # Extracting all the files of specific class
            with open(label_path, "r") as f:
                labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] == target_class]
            
            # Get random subset of target classes image ID's
            number_to_sample = round(amount_to_get * len(labels))
            print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split} in class {target_class}...")
            sampled_images = random.sample(labels, k=number_to_sample)
            
            # Apply full paths
            image_paths = [Path(str(images_path / sample_image) + ".jpg") for sample_image in sampled_images]
            class_images_path.extend(image_paths)
        
        label_splits[data_split] = class_images_path
    
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
    Loads and prepares the training and testing datasets for a machine learning model using the ImageFolder
    dataset class from `torchvision`. The datasets are transformed, batched, and returned as DataLoader objects 
    for easy iteration during training and evaluation. The function also returns the class labels found in the 
    training dataset.

    Args:
        train_path (str): The path to the directory containing the training images, organized in subdirectories 
                          where each subdirectory represents a class.
        test_path (str): The path to the directory containing the testing images, organized in the same way 
                         as the training images.
        batch_size (int): The number of samples to include in each batch for both the training and testing 
                          dataloaders.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
            - A DataLoader for the training dataset (`train_dataloader`).
            - A DataLoader for the testing dataset (`test_dataloader`).
            - A list of class labels (`class_labels`) as found in the training dataset.

    Example:
        train_loader, test_loader, class_labels = get_dataloaders(
            train_path='./data/train', 
            test_path='./data/test', 
            batch_size=32
        )

    Notes:
        - The training data undergoes resizing (224x224 pixels) and augmentation using `TrivialAugmentWide` 
          for improved model generalization.
        - The testing data is resized to 224x224 pixels and converted to tensors.
        - Both DataLoader objects are set to use `pin_memory=True` for improved data transfer speed on CUDA devices.
        - The function assumes the data is organized in subdirectories, with each subdirectory representing a class.
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
import os

import torch
import torchvision

from pathlib import Path
from typing import List, Tuple

def get_dataloaders(batch_size: int) -> Tuple[torch.utils.data.DataLoader, 
                                              torch.utils.data.DataLoader, 
                                              List[str]]:
    """
    Loads and returns the training and test dataloaders for the FashionMNIST dataset.

    This function downloads the FashionMNIST dataset (if not already present), 
    applies basic tensor transformations, and creates dataloaders for both the 
    training and testing sets. It also returns the class names for the dataset.

    Args:
        batch_size (int): The batch size for loading the data.
        num_workers (int): The number of workers to use for data loading.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
            - The training dataloader.
            - The test dataloader.
            - A list of class names in the dataset.
    """
    # Creating the data path
    data_path = Path('./data/')
    if not data_path.is_dir():
        os.mkdir(data_path)
    
    # Getting the datasets
    train_data = torchvision.datasets.FashionMNIST(root=data_path,
                                                   train=True,
                                                   transform=torchvision.transforms.ToTensor(),
                                                   download=True)
    
    test_data = torchvision.datasets.FashionMNIST(root=data_path,
                                                  train=False,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    
    # Getting the class names
    class_labels = train_data.classes
    
    # Loading the data to dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    
    return train_dataloader, test_dataloader, class_labels
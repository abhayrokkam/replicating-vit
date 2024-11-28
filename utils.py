import os

import torch
from torch.utils import tensorboard

from pathlib import Path
from datetime import datetime

def create_writer(model_name: str,
                  experiment_name: str) -> tensorboard.writer.SummaryWriter:
    """
    Creates a TensorBoard SummaryWriter for logging.

    Initializes a SummaryWriter with a log directory based on the current
    timestamp, model name, and experiment name.

    Args:
        model_name (str): The model name.
        experiment_name (str): The experiment name.

    Returns:
        tensorboard.writer.SummaryWriter: The configured SummaryWriter.
    """
    timestamp = datetime.now().strftime("%y-%m-%d")
    
    log_dir = os.path.join('runs', timestamp, model_name, experiment_name)
    
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    
    return tensorboard.writer.SummaryWriter(log_dir=log_dir)

def save_model(model: torch.nn.Module,
               model_name: str,
               target_dir: str) -> None:
    """
    Saves the model's state_dict to a specified directory.

    This function saves the model's `state_dict` (weights and biases) to a file
    in the specified target directory. The model is saved with the provided
    `model_name`, which must end in `.pth` or `.pt`.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name of the file where the model will be saved. Should end with `.pth` or `.pt`.
        target_dir (str): The directory where the model will be saved.

    Returns:
        None

    Raises:
        AssertionError: If `model_name` does not end with `.pth` or `.pt`.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "The argument 'model_name' should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
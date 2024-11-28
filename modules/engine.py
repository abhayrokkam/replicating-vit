import torch
import torchmetrics
import torch.utils.tensorboard

from tqdm.auto import tqdm

from typing import Dict, List

def train_epoch(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                loss_function: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_function: torchmetrics.Accuracy,
                device: torch.device) -> Dict[str, float]:
    """
    Trains the model for one epoch using the provided training data.

    This function performs one full pass through the training data, computing
    the loss, performing backpropagation, and updating the model parameters
    using the specified optimizer. It also computes the accuracy of the model
    on the training set.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        loss_function (torch.nn.Module): The loss function to be used for training.
        optimizer (torch.optim.Optimizer): The optimizer to update model weights.
        accuracy_function (torchmetrics.Accuracy): Metric to compute accuracy.
        device (torch.device): The device (CPU or GPU) on which to perform the training.

    Returns:
        Dict[str, float]: A dictionary containing the average training loss and accuracy 
                          for the epoch:
                          - 'train_loss' (float): The average loss over the training dataset.
                          - 'train_acc' (float): The average accuracy over the training dataset.    
    """
    # Model to device
    model = model.to(device)
    
    # Model to train mode
    model = model.train()

    # Track avg loss
    train_loss = 0
    train_acc = 0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        
        # Forward pass -> loss -> zero grad -> back prop -> gradient descent
        y_logits = model(X)
        loss = loss_function(y_logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accuracy
        accuracy_function = accuracy_function.to(device)
        y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1).squeeze()
        accuracy = accuracy_function(y_preds, y)
        
        # Accumulate
        train_loss += loss
        train_acc += accuracy

    # Average per batch
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)    
    return {'train_loss': train_loss.item(),
            'train_acc': train_acc.item()}

def test_epoch(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_function: torchmetrics.Accuracy,
               device: torch.device) -> Dict[str, float]:
    """
    Evaluates the model for one epoch using the provided test data.

    This function performs one full pass through the test data, computing the
    loss and accuracy of the model on the test set. The model is set to evaluation
    mode, and the loss and accuracy are accumulated and averaged over all test batches.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        loss_function (torch.nn.Module): The loss function to be used for evaluation.
        accuracy_function (torchmetrics.Accuracy): Metric to compute accuracy on the test set.
        device (torch.device): The device (CPU or GPU) on which to perform the evaluation.

    Returns:
        Dict[str, float]: A dictionary containing the average test loss and accuracy
                          for the epoch:
                          - 'test_loss' (float): The average loss over the test dataset.
                          - 'test_acc' (float): The average accuracy over the test dataset.    
    """
    # Model to device
    model = model.to(device)    

    # Set model to evaluation mode
    model = model.eval()

    # Track avg loss
    test_loss = 0
    test_acc = 0

    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        # With inference to save cuda memory
        with torch.inference_mode():
            # Loss
            y_logits = model(X)
            loss = loss_function(y_logits, y)
            
            # Accuracy
            accuracy_function = accuracy_function.to(device)
            y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1).squeeze()
            accuracy = accuracy_function(y_preds, y)
            
            # Accumulate
            test_loss += loss
            test_acc += accuracy

    # Average per batch
    with torch.inference_mode():
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    
    return {'test_loss': test_loss.item(),
            'test_acc': test_acc.item()}
    
def train(num_epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_function: torchmetrics.Accuracy,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List[float]]:
    """
    Trains and evaluates the model for a specified number of epochs.

    Trains the model on the training dataset and evaluates it on the test dataset at
    the end of each epoch. Logs training and test loss, accuracy to TensorBoard and 
    returns the metrics for each epoch.

    Args:
        num_epochs (int): Number of training epochs.
        model (torch.nn.Module): Model to train.
        train_dataloader (torch.utils.data.DataLoader): Training dataset loader.
        test_dataloader (torch.utils.data.DataLoader): Testing dataset loader.
        loss_function (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        accuracy_function (torchmetrics.Accuracy): Metric to compute accuracy.
        device (torch.device): Device to run the model on (CPU or GPU).
        writer (torch.utils.tensorboard.writer.SummaryWriter): TensorBoard writer for logging.

    Returns:
        Dict[str, List[float]]: A dictionary with lists of training and test metrics for each epoch:
            - 'train_loss': List of average training loss.
            - 'train_acc': List of training accuracy.
            - 'test_loss': List of average test loss.
            - 'test_acc': List of test accuracy.
    """
    results = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
    }
    
    for epoch in tqdm(range(num_epochs)):
            print("-"*100 + "\n")
            
            # Train for one epoch
            train_result = train_epoch(model=model,
                                        train_dataloader=train_dataloader,
                                        loss_function=loss_function,
                                        optimizer=optimizer,
                                        accuracy_function=accuracy_function,
                                        device=device)
            
            # Do testing after one epoch
            test_result = test_epoch(model=model,
                                    test_dataloader=test_dataloader,
                                    loss_function=loss_function,
                                    accuracy_function=accuracy_function,
                                    device=device)
            
            # Print results
            print(f"Epoch: {epoch+1}  |  Train Loss: {train_result['train_loss']:.2f}  |  Test Loss: {test_result['test_loss']:.2f}  |  Train Accuracy: {train_result['train_acc']:.2f}  |  Test Accuracy: {test_result['test_acc']:.2f}")
            
            # Track results
            results['train_loss'].append(train_result['train_loss'])
            results['train_acc'].append(train_result['train_acc'])
            results['test_loss'].append(test_result['test_loss'])
            results['test_acc'].append(test_result['test_acc'])
            
            # Using tensorboard writer for result tracking
            writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict={'train_loss': train_result['train_loss'],
                                                'test_loss': test_result['test_loss']},
                            global_step=epoch)
            
            writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={'train_acc': train_result['train_acc'],
                                                'test_acc': test_result['test_acc']},
                            global_step=epoch)
            
            # Empty cuda cache for memory management
            torch.cuda.empty_cache()
            
    writer.close()
    
    return results
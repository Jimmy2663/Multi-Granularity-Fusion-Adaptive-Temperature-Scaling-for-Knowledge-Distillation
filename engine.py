import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from timeit import default_timer as timer

from utils import (
    calculate_topk_accuracy,
    calculate_comprehensive_metrics,
    format_time,
    plot_training_curves,
    plot_top1_top3_comparison,
    generate_confusion_matrix,
    generate_classification_report,
    save_training_metrics,
    save_detailed_metrics,
    save_results_to_json,
    save_model,
    save_best_model
)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device
               ) -> dict:
    """Performs one epoch of training, returns comprehensive metrics.
    
    Passes the model through a full training epoch on the provided dataloader,
    computes and accumulates loss and accuracy.
    
    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader for the training dataset.
        loss_fn: Loss function to optimize.
        optimizer: Optimizer to update model parameters.
        device: Device to perform computations on ("cpu" or "cuda").
        
    Returns:
        A dictionary of training metrics for the epoch.
    """
    # Set model to training mode, enabling dropout and batchnorm
    model.train()
    
    # Initialize accumulators for loss and accuracy
    train_loss = 0.0
    all_preds, all_targets, all_proba = [], [], []
    total_top1_acc, total_top3_acc = 0.0, 0.0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Move batch data to computation device
        X, y = X.to(device), y.to(device)
        
        # Forward pass: compute model predictions
        y_pred = model(X)
        
        # Compute loss for the batch
        loss = loss_fn(y_pred, y)
        
        # Accumulate scalar loss value
        train_loss += loss.item()
        
        # Zero gradients before backward pass
        optimizer.zero_grad()
        
        # Backpropagate to compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Calculate top-k accuracies
        top1_acc, top3_acc = calculate_topk_accuracy(y_pred, y, topk=(1, 3))
        total_top1_acc += top1_acc.item()
        total_top3_acc += top3_acc.item()
        
        # Store predictions and targets for comprehensive metrics
        softmax_preds = torch.softmax(y_pred, dim=1)
        preds = torch.argmax(softmax_preds, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_proba.extend(softmax_preds.detach().cpu().numpy())
    
    # Compute average metrics
    num_batches = len(dataloader)
    avg_loss = train_loss / num_batches
    avg_top1_acc = total_top1_acc / num_batches
    avg_top3_acc = total_top3_acc / num_batches

    # Calculate comprehensive metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_proba = np.array(all_proba)
    
    num_classes = all_proba.shape[1] if len(all_proba) > 0 else 0
    comprehensive_metrics = calculate_comprehensive_metrics(
        all_targets, all_preds, all_proba, num_classes
    )
    
    # Combine all metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_top1_acc / 100.0,  # Convert from percentage to decimal
        'top1_accuracy': avg_top1_acc / 100.0,
        'top3_accuracy': avg_top3_acc / 100.0,
        **comprehensive_metrics
    }
    
    return metrics

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> dict:
    """Evaluates the model on one epoch, returns comprehensive metrics.
    
    Runs a forward pass in evaluation mode without gradient computations.
    
    Args:
        model: A PyTorch model to be evaluated.
        dataloader: A DataLoader for the validation or test dataset.
        loss_fn: Loss function for evaluation.
        device: Device to perform computations on.
        
    Returns:
        A dictionary of evaluation metrics for the dataset.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize accumulators for loss and accuracy
    test_loss = 0.0
    all_preds, all_targets, all_proba = [], [], []
    total_top1_acc, total_top3_acc = 0.0, 0.0
    
    # Disable gradient calculations for efficiency
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Move batch data to computation device
            X, y = X.to(device), y.to(device)
            
            # Forward pass: compute logits
            logits = model(X)
            
            # Compute loss for the batch
            loss = loss_fn(logits, y)
            
            # Accumulate scalar loss value
            test_loss += loss.item()
            
            # Calculate top-k accuracies
            top1_acc, top3_acc = calculate_topk_accuracy(logits, y, topk=(1, 3))
            total_top1_acc += top1_acc.item()
            total_top3_acc += top3_acc.item()
            
            # Store predictions and targets for comprehensive metrics
            softmax_preds = torch.softmax(logits, dim=1)
            preds = torch.argmax(softmax_preds, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_proba.extend(softmax_preds.cpu().numpy())
    
    # Compute average metrics
    num_batches = len(dataloader)
    avg_loss = test_loss / num_batches
    avg_top1_acc = total_top1_acc / num_batches
    avg_top3_acc = total_top3_acc / num_batches
    
    # Calculate comprehensive metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_proba = np.array(all_proba)
    
    num_classes = all_proba.shape[1] if len(all_proba) > 0 else 0
    comprehensive_metrics = calculate_comprehensive_metrics(
        all_targets, all_preds, all_proba, num_classes
    )
    
    # Combine all metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_top1_acc / 100.0,  # Convert from percentage to decimal
        'top1_accuracy': avg_top1_acc / 100.0,
        'top3_accuracy': avg_top3_acc / 100.0,
        **comprehensive_metrics
    }
    
    return metrics


def train_and_test(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  real_test_dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  epochs: int,
                  device: torch.device,
                  class_names: list = None,
                  save_dir: str = "plots") -> dict:
    """Trains and tests a PyTorch model with comprehensive reporting.
    
    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.
    
    Calculates, prints and stores evaluation metrics throughout.
    Generates plots, confusion matrix, classification report, and saves
    model checkpoints.
    
    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        real_test_dataloader: DataLoader for the final test dataset.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        class_names: List of class names for confusion matrix labeling.
        save_dir: Directory to save all plots and reports.
        
    Returns:
        A dictionary containing all training results, metrics, and file paths.
    """
    # Create main save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare checkpoint directory
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metric containers
    train_metrics_history, val_metrics_history = [], []
    
    # For plotting
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    train_top1_accuracies, train_top3_accuracies = [], []
    val_top1_accuracies, val_top3_accuracies = [], []
    
    best_metrics = {"epoch": 0, "val_loss": float("inf")}
    
    # Track best validation Top-1 accuracy
    best_top1 = 0.0

    # Move model to target device
    model.to(device)
    
    # Start overall training timer
    start_train_timer = timer()
    
    # Training and Validation loop
    for epoch in tqdm(range(epochs)):
        # Perform training step
        train_metrics = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        # Perform validation step
        val_metrics = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        # Record epoch metrics
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Extract metrics for plotting
        train_losses.append(train_metrics['loss'])
        train_accuracies.append(train_metrics['top1_accuracy'] * 100)  # Convert to percentage
        train_top1_accuracies.append(train_metrics['top1_accuracy'] * 100)
        train_top3_accuracies.append(train_metrics['top3_accuracy'] * 100)
        
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['top1_accuracy'] * 100)  # Convert to percentage
        val_top1_accuracies.append(val_metrics['top1_accuracy'] * 100)
        val_top3_accuracies.append(val_metrics['top3_accuracy'] * 100)
        
        # Console output per epoch
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Top-1: {train_metrics['top1_accuracy']*100:.2f}%, "
              f"Train Top-3: {train_metrics['top3_accuracy']*100:.2f}%, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Top-1: {val_metrics['top1_accuracy']*100:.2f}%, "
              f"Val Top-3: {val_metrics['top3_accuracy']*100:.2f}%")
        
        # Update best validation performance
        if val_metrics['loss'] < best_metrics["val_loss"]:
            best_metrics = {"epoch": epoch, "val_loss": val_metrics['loss']}
        
        # Save model when validation Top-1 improves
        if val_metrics['top1_accuracy'] > best_top1:
            best_top1 = val_metrics['top1_accuracy']
            best_path = os.path.join(save_dir, "best_model.pth")
            save_best_model(model, save_dir, "best_model.pth")
            print(f"New best model saved (val_Top-1: {best_top1*100:.2f}%) at {best_path}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 20 == 0:
            ckpt_name = f"model_epoch_{epoch+1}.pth"
            save_model(model, checkpoint_dir, ckpt_name)
    
    # End training timer
    end_train_timer = timer()
    training_time = end_train_timer - start_train_timer
    
    # Display total training time
    print(f"\nTotal training time: {format_time(training_time)}")
    
    # Start testing timer
    start_test_timer = timer()
    
    # Final test evaluation
    test_metrics = test_step(
        model=model,
        dataloader=real_test_dataloader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Display final test metrics
    print(f"\nFinal Test Loss: {test_metrics['loss']:.4f}, "
          f"Test Top-1: {test_metrics['top1_accuracy']*100:.2f}%, "
          f"Test Top-3: {test_metrics['top3_accuracy']*100:.2f}%")
    
    # End testing timer
    end_test_timer = timer()
    testing_time = end_test_timer - start_test_timer
    
    # Display total testing time
    print(f"Total testing time: {format_time(testing_time)}")
    
    # Generate plots and reports
    loss_curve_path, acc_curve_path = plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        save_dir=save_dir
    )
    
    # Generate Top-1 and Top-3 comparison plot
    top1_top3_path = plot_top1_top3_comparison(
        train_top1_accuracies=train_top1_accuracies,
        val_top1_accuracies=val_top1_accuracies,
        train_top3_accuracies=train_top3_accuracies,
        val_top3_accuracies=val_top3_accuracies,
        save_dir=save_dir
    )
    
    print(f"\nLoss curves saved to {loss_curve_path}")
    print(f"Accuracy curves saved to {acc_curve_path}")
    print(f"Top-1/Top-3 comparison saved to {top1_top3_path}")
    
    # Generate test confusion matrix and test classification report
    conf_matrix_path1, y_true1, y_pred1 = generate_confusion_matrix(
        model=model,
        dataloader=real_test_dataloader,
        class_names=class_names,
        device=device,
        name="Test_confusion_matrix.png",
        save_dir=save_dir
    )
    
    print(f"\nTest confusion matrix saved to {conf_matrix_path1}")
    
    cls_report, report_path1 = generate_classification_report(
        y_true=y_true1,
        y_pred=y_pred1,
        class_names=class_names,
        name="Test_classification_report.txt",
        save_dir=save_dir
    )
    
    print(f"\nTest Classification Report saved to {report_path1}")
    
    # Generate train confusion matrix and train classification report
    conf_matrix_path2, y_true2, y_pred2 = generate_confusion_matrix(
        model=model,
        dataloader=train_dataloader,
        class_names=class_names,
        device=device,
        name="Train_confusion_matrix.png",
        save_dir=save_dir
    )
    
    print(f"\nTrain confusion matrix saved to {conf_matrix_path2}")
    
    cls_report, report_path2 = generate_classification_report(
        y_true=y_true2,
        y_pred=y_pred2,
        class_names=class_names,
        name="Train_classification_report.txt",
        save_dir=save_dir
    )
    
    print(f"\nTrain Classification Report saved to {report_path2}")
    
    # Save training metrics
    metrics_path = save_training_metrics(
        train_metrics_history=train_metrics_history,
        val_metrics_history=val_metrics_history,
        test_metrics=test_metrics,
        training_time=training_time,
        save_dir=save_dir
    )
    
    print(f"\nTraining metrics saved to {metrics_path}")
    
    # Save detailed metrics
    detailed_path = save_detailed_metrics(
        train_metrics_history=train_metrics_history,
        val_metrics_history=val_metrics_history,
        test_metrics=test_metrics,
        training_time=training_time,
        save_dir=save_dir
    )
    
    print(f"\nDetailed metrics saved to {detailed_path}")
    
    # Compile and save results to JSON File
    results = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "train_top1_accuracies": train_top1_accuracies,
        "train_top3_accuracies": train_top3_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_top1_accuracies": val_top1_accuracies,
        "val_top3_accuracies": val_top3_accuracies,
        "test_loss": test_metrics['loss'],
        "test_top1_accuracy": test_metrics['top1_accuracy'] * 100,
        "test_top3_accuracy": test_metrics['top3_accuracy'] * 100,
        "training_time": training_time,
        "plots": {
            "loss_curve": loss_curve_path,
            "accuracy_curve": acc_curve_path,
            "top1_top3_comparison": top1_top3_path,
            "Test_confusion_matrix": conf_matrix_path1,
            "Train_confusion_matrix": conf_matrix_path2
        },
        "reports": {
            "Test_classification_report": report_path1,
            "Train_classification_report": report_path2,
            "training_metrics": metrics_path,
            "detailed_metrics": detailed_path
        }
    }
    
    json_path = save_results_to_json(results, save_dir)
    print(f"\nResults saved to {json_path}")
    
    # Return the compiled results dictionary
    return results


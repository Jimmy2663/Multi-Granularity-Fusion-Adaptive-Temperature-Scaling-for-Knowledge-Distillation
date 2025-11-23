"""
Contains various utility functions for PyTorch model used during training and testing .
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def save_model(model: torch.nn.Module,
               target_dir: str | Path,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to (str or Path).
        model_name: A filename for the saved model. Should include
          either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                   target_dir="models",
                   model_name="05_going_modular_tingvgg_model.pth")
    """
    # Convert target_dir to Path if it isn't already
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def calculate_topk_accuracy(output, target, topk=(1, 3)):
    """Calculates top-k accuracy for given k values.
    
    Args:
        output: Model predictions (logits)
        target: Ground truth labels
        topk: Tuple of k values for which to compute accuracy
        
    Returns:
        List of top-k accuracies for each k in topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # Get the top k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # Transpose to shape [k, batch_size]
        
        # Check if the target is in the top k predictions
        # Expand target to same shape as pred for comparison
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            # Count number of correct predictions in top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Calculate accuracy as percentage
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, num_classes):
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for each class
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    # Calculate precision, recall, f1 with different averaging methods
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Calculate AUC metrics if we have probability scores
    auc_metrics = {}
    if y_pred_proba is not None and y_pred_proba.shape[1] > 1:
        # One-hot encode the true labels for multi-class AUC calculation
        y_true_onehot = np.zeros((len(y_true), num_classes))
        for i, val in enumerate(y_true):
            y_true_onehot[i, val] = 1
            
        try:
            auc_micro = roc_auc_score(y_true_onehot, y_pred_proba, average='micro')
            auc_macro = roc_auc_score(y_true_onehot, y_pred_proba, average='macro')
            auc_weighted = roc_auc_score(y_true_onehot, y_pred_proba, average='weighted')
            
            auc_metrics = {
                'auc_micro': auc_micro,
                'auc_macro': auc_macro,
                'auc_weighted': auc_weighted
            }
        except Exception:
            # Fallback if AUC calculation fails
            auc_metrics = {
                'auc_micro': 0.0,
                'auc_macro': 0.0,
                'auc_weighted': 0.0
            }
    
    # Combine all metrics
    metrics = {
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        **auc_metrics
    }
    
    return metrics

def format_time(seconds):
    """Convert seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def plot_training_curves(train_losses,
                         val_losses,
                         train_accuracies,
                         val_accuracies,
                         save_dir: str = "plots") -> tuple:
    """Plots and saves loss and accuracy curves over epochs.
    
    Generates line plots for training vs. validation loss and accuracy.
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_accuracies: List of training accuracies per epoch.
        val_accuracies: List of validation accuracies per epoch.
        save_dir: Directory in which to save plot images.
        
    Returns:
        Tuple of file paths (loss_curve_path, accuracy_curve_path).
    """
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='red')
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    loss_curve_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label="Train Accuracy", color='blue')
    plt.plot(val_accuracies, label="Validation Accuracy", color='red')
    plt.title("Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    acc_curve_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return paths to saved plot images
    return loss_curve_path, acc_curve_path

def plot_top1_top3_comparison(train_top1_accuracies,
                              val_top1_accuracies,
                              train_top3_accuracies,
                              val_top3_accuracies,
                              save_dir: str = "plots") -> str:
    """Plot Top-1 and Top-3 accuracy comparison.
    
    Args:
        train_top1_accuracies: List of training Top-1 accuracies per epoch.
        val_top1_accuracies: List of validation Top-1 accuracies per epoch.
        train_top3_accuracies: List of training Top-3 accuracies per epoch.
        val_top3_accuracies: List of validation Top-3 accuracies per epoch.
        save_dir: Directory in which to save plot image.
        
    Returns:
        Path to the saved plot image.
    """
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    epochs = range(len(train_top1_accuracies))
    
    # Plot lines with appropriate styles
    plt.plot(epochs, train_top1_accuracies, 'b-', label='Train Top-1', linewidth=2)
    plt.plot(epochs, val_top1_accuracies, 'r-', label='Val Top-1', linewidth=2)
    plt.plot(epochs, train_top3_accuracies, 'b--', label='Train Top-3', linewidth=2)
    plt.plot(epochs, val_top3_accuracies, 'r--', label='Val Top-3', linewidth=2)
    
    # Add title, labels, legend, grid
    plt.title('Top-1 and Top-3 Accuracies', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Save plot
    plot_path = os.path.join(save_dir, "top1_top3_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_confusion_matrix(model,
                             dataloader,
                             class_names,
                             device,
                             name,
                             save_dir: str = "plots") -> tuple:
    """Generates and saves a confusion matrix heatmap.
    
    Computes true vs. predicted labels and visualizes them in a heatmap.
    
    Args:
        model: A PyTorch model to evaluate on the dataset.
        dataloader: DataLoader for the dataset to compute the matrix.
        class_names: List of class names for axis labels.
        device: Computation device.
        name: Name for the output file.
        save_dir: Directory to save the heatmap image.
        
    Returns:
        Tuple (conf_matrix_path, all_targets, all_predictions).
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists for labels
    all_preds, all_targets = [], []
    
    # Disable gradient calculations for efficiency
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Move batch data to computation device
            X, y = X.to(device), y.to(device)
            
            # Forward pass: compute logits
            logits = model(X)
            
            # Determine predicted classes
            preds = logits.argmax(dim=1)
            
            # Collect predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot and save heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    conf_matrix_path = os.path.join(save_dir, name)
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return path and raw label lists
    return conf_matrix_path, all_targets, all_preds

def generate_classification_report(y_true,
                                  y_pred,
                                  class_names,
                                  name,
                                  save_dir: str = "plots") -> tuple:
    """Generates and saves a text classification report.
    
    Creates precision/recall/f1 metrics per class and writes to file.
    
    Args:
        y_true: Ground truth labels list.
        y_pred: Predicted labels list.
        class_names: Names of classes for report formatting.
        name: Name for the output file.
        save_dir: Directory to save the text report.
        
    Returns:
        Tuple (report_string, report_path).
    """
    # Compute text report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, name)
    
    # Write report to file
    with open(report_path, "w") as f:
        f.write(report)
    
    # Return report content and file path
    return report, report_path

def save_training_metrics(train_metrics_history,
                         val_metrics_history,
                         test_metrics,
                         training_time,
                         save_dir: str = "plots") -> str:
    """Saves epoch-wise training and final test metrics to text file.
    
    Writes loss and accuracy for each epoch plus final test metrics.
    
    Args:
        train_metrics_history: List of dictionaries with training metrics per epoch.
        val_metrics_history: List of dictionaries with validation metrics per epoch.
        test_metrics: Dictionary with final test metrics.
        training_time: Total training time in seconds.
        save_dir: Directory to save the metrics file.
        
    Returns:
        Path to the saved training metrics file.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    path = os.path.join(save_dir, "training_metrics.txt")
    
    # Write metrics to file
    with open(path, "w") as f:
        f.write("Epoch-wise Training and Validation Metrics:\n")
        
        for i in range(len(train_metrics_history)):
            f.write(f"Epoch {i+1}:\n")
            f.write(f"  Train Loss: {train_metrics_history[i]['loss']:.4f}, "
                   f"Train Top-1: {train_metrics_history[i]['top1_accuracy']*100:.2f}%, "
                   f"Train Top-3: {train_metrics_history[i]['top3_accuracy']*100:.2f}%\n")
            f.write(f"  Val Loss: {val_metrics_history[i]['loss']:.4f}, "
                   f"Val Top-1: {val_metrics_history[i]['top1_accuracy']*100:.2f}%, "
                   f"Val Top-3: {val_metrics_history[i]['top3_accuracy']*100:.2f}%\n\n")
        
        f.write(f"Final Test Loss: {test_metrics['loss']:.4f}, "
               f"Test Top-1: {test_metrics['top1_accuracy']*100:.2f}%, "
               f"Test Top-3: {test_metrics['top3_accuracy']*100:.2f}%\n")
        f.write(f"Total training time: {format_time(training_time)}\n")
    
    # Return file path
    return path

def format_metric_value(value):
    """
    Universal formatter for metric values.
    Handles floats, ints, lists, arrays, dicts, and other types.
    
    Args:
        value: Any value to format
        
    Returns:
        Formatted string representation of the value
    """
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    elif isinstance(value, (int, np.integer)):
        return f"{value}"
    elif isinstance(value, (list, tuple)):
        # Format lists/tuples element by element
        formatted_items = []
        for item in value:
            if isinstance(item, (float, np.floating)):
                formatted_items.append(f"{item:.4f}")
            else:
                formatted_items.append(str(item))
        return "[" + ", ".join(formatted_items) + "]"
    elif isinstance(value, np.ndarray):
        # Convert numpy array to formatted list
        if value.dtype in [np.float32, np.float64]:
            return "[" + ", ".join([f"{v:.4f}" for v in value.flat]) + "]"
        else:
            return "[" + ", ".join([str(v) for v in value.flat]) + "]"
    elif isinstance(value, dict):
        # Format dictionary nicely
        items = []
        for k, v in value.items():
            items.append(f"{k}: {format_metric_value(v)}")
        return "{" + ", ".join(items) + "}"
    else:
        # Fallback for any other type
        return str(value)


def save_detailed_metrics(train_metrics_history,
                         val_metrics_history,
                         test_metrics,
                         training_time,
                         save_dir: str = "plots") -> str:
    """Saves detailed metrics to a text file.
    
    Writes comprehensive metrics for training, validation, and test.
    Works with all data types: floats, ints, lists, arrays, dicts, etc.
    
    Args:
        train_metrics_history: List of dictionaries with training metrics per epoch.
        val_metrics_history: List of dictionaries with validation metrics per epoch.
        test_metrics: Dictionary with final test metrics.
        training_time: Total training time in seconds.
        save_dir: Directory to save the metrics file.
        
    Returns:
        Path to the saved detailed metrics file.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    path = os.path.join(save_dir, "detailed_metrics.txt")
    
    # Write metrics to file
    with open(path, "w") as f:
        # ================================================================
        # TRAINING METRICS (Final Epoch)
        # ================================================================
        f.write("=" * 70 + "\n")
        f.write("TRAINING METRICS (Final Epoch)\n")
        f.write("=" * 70 + "\n\n")
        
        if len(train_metrics_history) > 0:
            final_train_metrics = train_metrics_history[-1]
            for metric, value in final_train_metrics.items():
                formatted_value = format_metric_value(value)
                f.write(f"  {metric:<40} : {formatted_value}\n")
        else:
            f.write("  No training metrics available\n")
        
        # ================================================================
        # VALIDATION METRICS (Best Epoch)
        # ================================================================
        f.write("\n" + "=" * 70 + "\n")
        f.write("VALIDATION METRICS (Best Epoch)\n")
        f.write("=" * 70 + "\n\n")
        
        if len(val_metrics_history) > 0:
            # Find best validation epoch (lowest loss)
            best_val_epoch = min(range(len(val_metrics_history)), 
                                key=lambda i: val_metrics_history[i].get('loss', float('inf')))
            best_val_metrics = val_metrics_history[best_val_epoch]
            
            f.write(f"  Best epoch: {best_val_epoch + 1}\n\n")
            
            for metric, value in best_val_metrics.items():
                formatted_value = format_metric_value(value)
                f.write(f"  {metric:<40} : {formatted_value}\n")
        else:
            f.write("  No validation metrics available\n")
        
        # ================================================================
        # TEST METRICS
        # ================================================================
        f.write("\n" + "=" * 70 + "\n")
        f.write("TEST METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        if test_metrics:
            for metric, value in test_metrics.items():
                formatted_value = format_metric_value(value)
                f.write(f"  {metric:<40} : {formatted_value}\n")
        else:
            f.write("  No test metrics available\n")
        
        # ================================================================
        # TRAINING TIME
        # ================================================================
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRAINING TIME\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"  Total training time                : {format_time(training_time)}\n")
        
        if len(train_metrics_history) > 0:
            avg_time_per_epoch = training_time / len(train_metrics_history)
            f.write(f"  Average time per epoch             : {avg_time_per_epoch:.2f} seconds\n")
            f.write(f"  Total epochs                       : {len(train_metrics_history)}\n")
        
    # Return file path
    return path

def save_results_to_json(results: dict,
                        save_dir: str = "plots") -> str:
    """Dumps the results dictionary to a JSON file.
    
    Serializes all training, evaluation, and artifact paths.
    
    Args:
        results: Dictionary of results to be saved.
        save_dir: Directory to save the JSON file.
        
    Returns:
        Path to the saved JSON results file.
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    path = os.path.join(save_dir, "results.json")
    
    # Serialize dictionary to JSON
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Return file path
    return path


# save_model_summary.
def save_model_summary(model, input_size, file_path, use_torchinfo=True, device="cpu"):
    """
    Save a summary of a PyTorch model to a text file.

    Args:
        model: torch.nn.Module, the model to summarize.
        input_size: tuple or list, the input size (e.g., [3, 224, 224] for images).
                    For torchsummary, do NOT include the batch dimension.
        file_path: str, path to save the summary text file.
        use_torchinfo: bool, if True uses torchinfo, else uses torchsummary.
        device: str or torch.device, device for summary ("cpu" or "cuda").
    """
    device = str(device)  # Ensure device is a string

    if use_torchinfo:
        try:
            from torchinfo import summary
            with open(file_path, "w") as f:
                # torchinfo expects input_size as a list/tuple, including batch if needed
                print(summary(model, input_size=input_size, device=device), file=f)
        except ImportError:
            raise ImportError("Please install torchinfo: pip install torchinfo")
    else:
        try:
            from torchsummary import summary_string
            input_size_no_batch = input_size[1:] if len(input_size) > 3 else input_size
            report, _ = summary_string(model, input_size=input_size_no_batch, device=device)
            with open(file_path, "w") as f:
                f.write(report)
        except ImportError:
            raise ImportError("Please install torchsummary: pip install torchsummary")

# Fuction to save best model while training same as save_model but doesnt print model saved status after done         
def save_best_model(model: torch.nn.Module,
               target_dir: str | Path,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to (str or Path).
        model_name: A filename for the saved model. Should include
          either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                   target_dir="models",
                   model_name="05_going_modular_tingvgg_model.pth")
    """
    # Convert target_dir to Path if it isn't already
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(), f=model_save_path)

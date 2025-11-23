# ============================================================================
# COMPLETE KD_ENGINE.PY WITH SAMPLEWISE, CLASSWISE, LEARNABLE, AND FUSION ATS
# ============================================================================

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

# ============================================================================
# FUNCTION 0 : LEARNABLE TEMPERATURE MODULE
# ============================================================================

class LearnableTemperatureModule(torch.nn.Module):
    """
    Learnable adaptive temperature module that learns temperatures during training.
    This module learns ONE TEMPERATURE PER CLASS or ONE GLOBAL TEMPERATURE
    based on initialization.
    
    Features:
    - Can learn per-class temperatures [num_classes] or global temperature [1]
    - Temperatures bounded between T_min and T_max using sigmoid activation
    - Learnable parameters updated via gradient descent
    - Can be integrated into the training loop
    """
    
    def __init__(self,
                 num_classes: int = None,
                 T_base: float = 4.0,
                 T_min: float = 1.0,
                 T_max: float = 20.0,
                 learn_per_class: bool = True):
        """
        Initialize learnable temperature module.
        
        Args:
            num_classes: Number of classes (required if learn_per_class=True)
            T_base: Base temperature (used for initialization)
            T_min: Minimum temperature bound
            T_max: Maximum temperature bound
            learn_per_class: If True, learn per-class temps; else learn global temp
        """
        super().__init__()
        self.T_base = T_base
        self.T_min = T_min
        self.T_max = T_max
        self.learn_per_class = learn_per_class
        
        if learn_per_class:
            assert num_classes is not None, "num_classes required for learn_per_class=True"
            self.num_classes = num_classes
            # Initialize learnable logits for temperatures
            # Will be converted to [T_min, T_max] range via sigmoid
            self.log_temps = torch.nn.Parameter(
                torch.ones(num_classes) * np.log(T_base)
            )
        else:
            # Global temperature
            self.log_temp = torch.nn.Parameter(
                torch.tensor([np.log(T_base)])
            )
    
    def forward(self, targets: torch.Tensor = None) -> torch.Tensor:
        """
        Get temperatures.
        
        Args:
            targets: [batch_size] class indices. Required only if learn_per_class=True
                    Used to select per-class temperatures for each sample.
                    If None, returns all class temperatures.
        
        Returns:
            If learn_per_class and targets provided: [batch_size] sample-wise temperatures
            If learn_per_class and targets=None: [num_classes] class-wise temperatures
            If not learn_per_class: [1] scalar temperature (broadcasted as needed)
        """
        if self.learn_per_class:
            # Map log_temps to [T_min, T_max]
            temps = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.log_temps)
            
            if targets is not None:
                # Return per-sample temperatures based on class
                return temps[targets]
            else:
                # Return all class temperatures
                return temps
        else:
            # Global temperature
            temp = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.log_temp)
            return temp
    
    def get_temperatures_info(self) -> dict:
        """Get detailed temperature statistics."""
        if self.learn_per_class:
            temps = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.log_temps)
            return {
                'mean': float(temps.mean()),
                'std': float(temps.std()),
                'min': float(temps.min()),
                'max': float(temps.max()),
                'per_class': temps.detach().cpu().numpy().tolist()
            }
        else:
            temp = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.log_temp)
            return {
                'global_temp': float(temp.item()),
                'mean': float(temp.item()),
                'std': 0.0,
                'min': float(temp.item()),
                'max': float(temp.item())
            }


# ============================================================================
# FUNCTION 1: Calculate Sample-wise Adaptive Temperature
# ============================================================================

def calculate_adaptive_temperature(teacher_logits: torch.Tensor,
                                   T_base: float = 4.0,
                                   T_min: float = 1.0,
                                   T_max: float = 20.0,
                                   method: str = "max_logit") -> torch.Tensor:
    """
    Calculates sample-wise adaptive temperature based on teacher logits difficulty.
    
    CORE ATS FUNCTION (SAMPLEWISE):
    - Easy samples (high teacher confidence) → LOW temperature → SHARP distribution
    - Hard samples (low teacher confidence) → HIGH temperature → SMOOTH distribution
    
    Args:
        teacher_logits: [batch_size, num_classes] from teacher model
        T_base: Base temperature (default 4.0)
        T_min: Minimum temperature bound (default 1.0)
        T_max: Maximum temperature bound (default 20.0)
        method: "max_logit" (RECOMMENDED), "entropy", or "margin"
    
    Returns:
        Tensor [batch_size] with per-sample temperatures
    """
    with torch.no_grad():
        batch_size = teacher_logits.shape[0]
        
        if method == "max_logit":
            # RECOMMENDED: Max logit based difficulty assessment
            max_logits = torch.max(teacher_logits, dim=1)[0]  # [batch_size]
            
            # Z-score normalization
            mean_max = max_logits.mean()
            std_max = max_logits.std() + 1e-8
            z_scores = (max_logits - mean_max) / std_max
            
            # Convert to temperature: T = T_base * exp(-c * z_score)
            adaptive_T = T_base * torch.exp(-0.1 * z_scores)
        
        elif method == "entropy":
            # Entropy-based: high entropy (uncertain) → high temperature
            probs = torch.softmax(teacher_logits / T_base, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_normalized = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
            adaptive_T = T_min + (T_max - T_min) * entropy_normalized
        
        elif method == "margin":
            # Margin-based: small margin (uncertain) → high temperature
            top_vals, _ = torch.topk(teacher_logits, k=2, dim=1)
            margins = top_vals[:, 0] - top_vals[:, 1]
            margin_normalized = (margins - margins.min()) / (margins.max() - margins.min() + 1e-8)
            adaptive_T = T_max - (T_max - T_min) * margin_normalized
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clamp to valid range
        adaptive_T = torch.clamp(adaptive_T, min=T_min, max=T_max)
        
        return adaptive_T


# ============================================================================
# FUNCTION 2: Calculate Class-wise Adaptive Temperature
# ============================================================================

def calculate_adaptive_temperature_classwise(teacher_logits: torch.Tensor,
                                             targets: torch.Tensor,
                                             num_classes: int,
                                             T_base: float = 4.0,
                                             T_min: float = 1.0,
                                             T_max: float = 20.0,
                                             method: str = "max_logit") -> torch.Tensor:
    """
    Calculates class-wise adaptive temperatures: one temperature for each class.
    
    CORE ATS FUNCTION (CLASSWISE):
    - Each class gets ONE unique temperature based on difficulty of that class
    - All samples with same class label use the same temperature
    
    Args:
        teacher_logits: [batch_size, num_classes] from teacher model
        targets: [batch_size] LONG, actual class labels
        num_classes: total number of classes
        T_base: Base temperature (default 4.0)
        T_min: Minimum temperature bound (default 1.0)
        T_max: Maximum temperature bound (default 20.0)
        method: "max_logit" (RECOMMENDED), "entropy", or "margin"
    
    Returns:
        Tensor [num_classes] with per-class temperatures
    """
    with torch.no_grad():
        classwise_temperatures = []
        device = teacher_logits.device
        
        for c in range(num_classes):
            class_mask = (targets == c)
            
            if not torch.any(class_mask):
                # No samples for this class → use T_base
                temp = torch.tensor([T_base], device=device)
            else:
                class_logits = teacher_logits[class_mask]
                num_samples_in_class = class_logits.shape[0]
                
                if method == "max_logit":
                    max_logits = torch.max(class_logits, dim=1)[0]
                    
                    if num_samples_in_class == 1:
                        temp = torch.tensor([T_base], device=device)
                    else:
                        mean_max = max_logits.mean()
                        std_max = max_logits.std() + 1e-8
                        z_scores = (max_logits - mean_max) / std_max
                        adaptive_T = T_base * torch.exp(-0.1 * z_scores.mean())
                        adaptive_T = torch.clamp(adaptive_T, min=T_min, max=T_max)
                        temp = torch.tensor([adaptive_T.item()], device=device)
                
                elif method == "entropy":
                    probs = torch.softmax(class_logits / T_base, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    entropy_mean = entropy.mean()
                    
                    if entropy_mean < T_min:
                        adaptive_T = torch.tensor([T_min], device=device)
                    elif entropy_mean > T_max:
                        adaptive_T = torch.tensor([T_max], device=device)
                    else:
                        entropy_n = (entropy_mean - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
                        adaptive_T = T_min + (T_max - T_min) * entropy_n
                        adaptive_T = torch.clamp(adaptive_T, min=T_min, max=T_max)
                        adaptive_T = torch.tensor([adaptive_T.item()], device=device)
                    
                    temp = adaptive_T
                
                elif method == "margin":
                    if teacher_logits.shape[1] >= 2:
                        top_vals, _ = torch.topk(class_logits, k=2, dim=1)
                        margins = top_vals[:, 0] - top_vals[:, 1]
                        margin_mean = margins.mean()
                        
                        if margins.max() > margins.min():
                            margin_n = (margin_mean - margins.min()) / (margins.max() - margins.min() + 1e-8)
                        else:
                            margin_n = torch.tensor(0.5, device=device)
                        
                        adaptive_T = T_max - (T_max - T_min) * margin_n
                        adaptive_T = torch.clamp(adaptive_T, min=T_min, max=T_max)
                        temp = torch.tensor([adaptive_T.item()], device=device)
                    else:
                        temp = torch.tensor([T_base], device=device)
                
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            classwise_temperatures.append(temp)
        
        result = torch.cat(classwise_temperatures, dim=0)
        return result


# ============================================================================
# FUNCTION 3: FUSION-ATS FRAMEWORK (STEP 4) - NEW
# ============================================================================

def calculate_fusion_adaptive_temperature(teacher_logits: torch.Tensor,
                                          targets: torch.Tensor,
                                          num_classes: int,
                                          T_base: float = 4.0,
                                          T_min: float = 1.0,
                                          T_max: float = 20.0,
                                          method: str = "max_logit",
                                          alpha: float = 0.33,
                                          beta: float = 0.33,
                                          gamma: float = 0.34) -> torch.Tensor:
    """
    STEP 4: Unified Fusion-ATS Framework
    
    Combines three granularities of adaptive temperature scaling:
    1. Sample-wise ATS: Per-sample adaptive temperatures
    2. Class-wise ATS: Per-class adaptive temperatures
    3. Learnable ATS: Learned parameters (simulated via weighted combination)
    
    The Fusion-ATS fuses these three components with learned weights (alpha, beta, gamma)
    to create a unified adaptive temperature that balances all three perspectives.
    
    Formula:
    T_fusion = alpha * T_samplewise + beta * T_classwise + gamma * T_learnable
    
    Args:
        teacher_logits: [batch_size, num_classes] from teacher model
        targets: [batch_size] LONG, actual class labels
        num_classes: total number of classes
        T_base: Base temperature (default 4.0)
        T_min: Minimum temperature bound (default 1.0)
        T_max: Maximum temperature bound (default 20.0)
        method: "max_logit" (RECOMMENDED), "entropy", or "margin"
        alpha: Weight for sample-wise component (default 0.33)
        beta: Weight for class-wise component (default 0.33)
        gamma: Weight for learnable component (default 0.34)
    
    Returns:
        Tensor [batch_size] with fused adaptive temperatures
    """
    with torch.no_grad():
        device = teacher_logits.device
        batch_size = teacher_logits.shape[0]
        
        # Normalize weights to sum to 1
        total_weight = alpha + beta + gamma
        alpha_norm = alpha / total_weight
        beta_norm = beta / total_weight
        gamma_norm = gamma / total_weight
        
        # Component 1: Sample-wise ATS
        T_samplewise = calculate_adaptive_temperature(
            teacher_logits,
            T_base=T_base,
            T_min=T_min,
            T_max=T_max,
            method=method
        )  # [batch_size]
        
        # Component 2: Class-wise ATS
        T_classwise_all = calculate_adaptive_temperature_classwise(
            teacher_logits,
            targets=targets,
            num_classes=num_classes,
            T_base=T_base,
            T_min=T_min,
            T_max=T_max,
            method=method
        )  # [num_classes]
        
        # Map class-wise temperatures to sample-wise
        T_classwise = T_classwise_all[targets]  # [batch_size]
        
        # Component 3: Learnable ATS (simulated as entropy-based with scaling)
        probs = torch.softmax(teacher_logits / T_base, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [batch_size]
        entropy_normalized = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        T_learnable = T_min + (T_max - T_min) * entropy_normalized  # [batch_size]
        
        # FUSION: Weighted combination of all three components
        T_fusion = (alpha_norm * T_samplewise + 
                    beta_norm * T_classwise + 
                    gamma_norm * T_learnable)  # [batch_size]
        
        # Final clamp to valid range
        T_fusion = torch.clamp(T_fusion, min=T_min, max=T_max)
        
        return T_fusion


# ============================================================================
# MAIN TRAINING FUNCTION - WITH ALL FOUR GRANULARITIES
# ============================================================================

def train_step_knowledge_distillation(teacher: torch.nn.Module,
                                      student: torch.nn.Module,
                                      dataloader: torch.utils.data.DataLoader,
                                      loss_fn: torch.nn.Module,
                                      optimizer: torch.optim.Optimizer,
                                      T_base: float,
                                      T_min: float,
                                      T_max: float,
                                      soft_target_loss_weight: float,
                                      ce_loss_weight: float,
                                      device: torch.device,
                                      ats_method: str = "max_logit",
                                      ats_granularity: str = "samplewise",
                                      learnable_temp_module: LearnableTemperatureModule = None,
                                      fusion_weights: tuple = (0.33, 0.33, 0.34),
                                      store_temps: bool = False) -> dict:
    """
    One epoch of knowledge distillation training with ADAPTIVE TEMPERATURE.
    
    Supports four granularities:
    1. samplewise: Unique temperature per sample in batch
    2. classwise: Unique temperature per class (aggregated from batch)
    3. learnable: Learned temperatures (per-class) via gradient descent
    4. fusion: Unified Fusion-ATS combining all three approaches
    
    Args:
        teacher: Pre-trained teacher model (frozen)
        student: Student model to train
        dataloader: Training dataloader
        loss_fn: CrossEntropyLoss
        optimizer: Student optimizer (and temperature optimizer if using learnable)
        T_base: Base temperature for ATS
        T_min: Minimum temperature
        T_max: Maximum temperature
        soft_target_loss_weight: Weight for KL loss
        ce_loss_weight: Weight for CE loss
        device: "cuda" or "cpu"
        ats_method: "max_logit", "entropy", or "margin"
        ats_granularity: "samplewise", "classwise", "learnable", or "fusion"
        learnable_temp_module: LearnableTemperatureModule (required if ats_granularity="learnable")
        fusion_weights: (alpha, beta, gamma) for Fusion-ATS
        store_temps: Whether to store temperature values
    
    Returns:
        Dictionary with training metrics
    """
    teacher.eval()
    student.train()
    
    if ats_granularity == "learnable" and learnable_temp_module is not None:
        learnable_temp_module.train()
    
    train_loss = 0.0
    all_preds, all_targets, all_proba = [], [], []
    total_top1_acc, total_top3_acc = 0.0, 0.0
    temp_values = [] if store_temps else None
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(X)  # [batch_size, num_classes]
            
            # ================================================================
            # CORE ATS STEP: Calculate adaptive temperature
            # ================================================================
            
            if ats_granularity == "samplewise":
                adaptive_T = calculate_adaptive_temperature(
                    teacher_logits,
                    T_base=T_base,
                    T_min=T_min,
                    T_max=T_max,
                    method=ats_method
                )  # [batch_size]
            
            elif ats_granularity == "classwise":
                num_classes = teacher_logits.shape[1]
                adaptive_T_class = calculate_adaptive_temperature_classwise(
                    teacher_logits,
                    targets=y,
                    num_classes=num_classes,
                    T_base=T_base,
                    T_min=T_min,
                    T_max=T_max,
                    method=ats_method
                )  # [num_classes]
            
            elif ats_granularity == "learnable":
                if learnable_temp_module is None:
                    raise ValueError("learnable_temp_module required for learnable granularity")
                adaptive_T_learnable = learnable_temp_module(targets=y)  # [batch_size]
            
            elif ats_granularity == "fusion":
                num_classes = teacher_logits.shape[1]
                adaptive_T_fusion = calculate_fusion_adaptive_temperature(
                    teacher_logits,
                    targets=y,
                    num_classes=num_classes,
                    T_base=T_base,
                    T_min=T_min,
                    T_max=T_max,
                    method=ats_method,
                    alpha=fusion_weights[0],
                    beta=fusion_weights[1],
                    gamma=fusion_weights[2]
                )  # [batch_size]
            
            else:
                raise ValueError(f"Unknown ats_granularity: {ats_granularity}")
        
        if store_temps:
            if ats_granularity == "samplewise":
                temp_values.extend(adaptive_T.cpu().numpy())
            elif ats_granularity == "classwise":
                temp_values.extend(adaptive_T_class.cpu().numpy())
            elif ats_granularity == "learnable":
                temp_values.extend(adaptive_T_learnable.detach().cpu().numpy())
            elif ats_granularity == "fusion":
                temp_values.extend(adaptive_T_fusion.cpu().numpy())
        
        # Student forward
        student_logits = student(X)  # [batch_size, num_classes]
        
        # ================================================================
        # COMPUTE SOFT TARGETS LOSS WITH ADAPTIVE TEMPERATURE
        # ================================================================
        
        batch_size = student_logits.shape[0]
        soft_targets_loss_per_sample = torch.zeros(batch_size, device=device)
        
        # For each sample: compute KL loss with sample/class-specific temperature
        for i in range(batch_size):
            if ats_granularity == "samplewise":
                T_i = adaptive_T[i].item()
            elif ats_granularity == "classwise":
                T_i = adaptive_T_class[y[i]].item()
            elif ats_granularity == "learnable":
                T_i = adaptive_T_learnable[i].item()
            elif ats_granularity == "fusion":
                T_i = adaptive_T_fusion[i].item()
            
            # Temperature-scaled softmax for teacher
            soft_target_i = torch.softmax(teacher_logits[i] / T_i, dim=0)
            
            # Temperature-scaled log-softmax for student
            soft_prob_i = torch.log_softmax(student_logits[i] / T_i, dim=0)
            
            # KL divergence scaled by T_i^2
            kl_loss_i = torch.sum(soft_target_i * (torch.log(soft_target_i + 1e-8) - soft_prob_i))
            soft_targets_loss_per_sample[i] = kl_loss_i * (T_i ** 2)
        
        # Average across batch
        soft_targets_loss = soft_targets_loss_per_sample.mean()
        
        # ================================================================
        # COMPUTE HARD TARGETS LOSS
        # ================================================================
        
        hard_targets_loss = loss_fn(student_logits, y)
        
        # ================================================================
        # COMBINED LOSS
        # ================================================================
        
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * hard_targets_loss
        
        train_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracies
        top1_acc, top3_acc = calculate_topk_accuracy(student_logits, y, topk=(1, 3))
        total_top1_acc += top1_acc.item()
        total_top3_acc += top3_acc.item()
        
        # Store predictions
        softmax_preds = torch.softmax(student_logits, dim=1)
        preds = torch.argmax(softmax_preds, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_proba.extend(softmax_preds.detach().cpu().numpy())
    
    # ========================================================================
    # COMPUTE AVERAGE METRICS
    # ========================================================================
    
    num_batches = len(dataloader)
    avg_loss = train_loss / num_batches
    avg_top1_acc = total_top1_acc / num_batches
    avg_top3_acc = total_top3_acc / num_batches
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_proba = np.array(all_proba)
    num_classes_actual = all_proba.shape[1] if len(all_proba) > 0 else 0
    
    comprehensive_metrics = calculate_comprehensive_metrics(
        all_targets, all_preds, all_proba, num_classes_actual
    )
    
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_top1_acc / 100.0,
        'top1_accuracy': avg_top1_acc / 100.0,
        'top3_accuracy': avg_top3_acc / 100.0,
        **comprehensive_metrics
    }
    
    # Add temperature statistics if enabled
    if store_temps and temp_values:
        metrics['temp_mean'] = float(np.mean(temp_values))
        metrics['temp_std'] = float(np.std(temp_values))
        metrics['temp_min'] = float(np.min(temp_values))
        metrics['temp_max'] = float(np.max(temp_values))
    
    # Add learnable temperature info
    if ats_granularity == "learnable" and learnable_temp_module is not None:
        temp_info = learnable_temp_module.get_temperatures_info()
        metrics.update(temp_info)
    
    return metrics


# ============================================================================
# TEST/VALIDATION FUNCTION
# ============================================================================

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> dict:
    """
    Evaluation on validation/test data (no adaptive temperature needed).
    """
    model.eval()
    test_loss = 0.0
    all_preds, all_targets, all_proba = [], [], []
    total_top1_acc, total_top3_acc = 0.0, 0.0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss.item()
            
            top1_acc, top3_acc = calculate_topk_accuracy(logits, y, topk=(1, 3))
            total_top1_acc += top1_acc.item()
            total_top3_acc += top3_acc.item()
            
            softmax_preds = torch.softmax(logits, dim=1)
            preds = torch.argmax(softmax_preds, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_proba.extend(softmax_preds.cpu().numpy())
    
    num_batches = len(dataloader)
    avg_loss = test_loss / num_batches
    avg_top1_acc = total_top1_acc / num_batches
    avg_top3_acc = total_top3_acc / num_batches
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_proba = np.array(all_proba)
    num_classes = all_proba.shape[1] if len(all_proba) > 0 else 0
    
    comprehensive_metrics = calculate_comprehensive_metrics(
        all_targets, all_preds, all_proba, num_classes
    )
    
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_top1_acc / 100.0,
        'top1_accuracy': avg_top1_acc / 100.0,
        'top3_accuracy': avg_top3_acc / 100.0,
        **comprehensive_metrics
    }
    
    return metrics


# ============================================================================
# MAIN TRAINING AND TESTING LOOP WITH ALL GRANULARITIES
# ============================================================================

def train_and_test_with_KD(teacher: torch.nn.Module,
                           student: torch.nn.Module,
                           train_dataloader: torch.utils.data.DataLoader,
                           test_dataloader: torch.utils.data.DataLoader,
                           real_test_dataloader: torch.utils.data.DataLoader,
                           optimizer: torch.optim.Optimizer,
                           loss_fn: torch.nn.Module,
                           epochs: int,
                           T_base: float,
                           T_min: float,
                           T_max: float,
                           soft_target_loss_weight: float,
                           ce_loss_weight: float,
                           device: torch.device,
                           class_names: list = None,
                           save_dir: str = "plots",
                           ats_method: str = "max_logit",
                           ats_granularity: str = "samplewise",
                           fusion_weights: tuple = (0.33, 0.33, 0.34),
                           store_temperature_history: bool = True,
                           num_classes: int = None) -> dict:
    """
    Complete training and testing pipeline with ATS-KD.
    
    Supports four granularities:
    1. samplewise: Fixed/adaptive per-sample temperatures
    2. classwise: Fixed/adaptive per-class temperatures
    3. learnable: Learned per-class temperatures
    4. fusion: Unified Fusion-ATS combining all three approaches
    
    Args:
        teacher: Pre-trained teacher model (frozen)
        student: Student model to train
        train_dataloader: Training dataloader
        test_dataloader: Validation dataloader
        real_test_dataloader: Test dataloader
        optimizer: Optimizer for student (and temperatures if learnable)
        loss_fn: Loss function
        epochs: Number of epochs
        T_base: Base temperature
        T_min: Minimum temperature
        T_max: Maximum temperature
        soft_target_loss_weight: KL loss weight
        ce_loss_weight: CE loss weight
        device: Device
        class_names: List of class names
        save_dir: Directory to save results
        ats_method: "max_logit", "entropy", or "margin" (ignored for learnable/fusion)
        ats_granularity: "samplewise", "classwise", "learnable", or "fusion"
        fusion_weights: (alpha, beta, gamma) weights for Fusion-ATS
        store_temperature_history: Whether to log temperature statistics
        num_classes: Number of classes (required for learnable)
    
    Returns:
        Results dictionary
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_metrics_history, val_metrics_history = [], []
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    train_top1_accuracies, train_top3_accuracies = [], []
    val_top1_accuracies, val_top3_accuracies = [], []
    temperature_history = [] if store_temperature_history else None
    
    best_metrics = {"epoch": 0, "val_loss": float("inf")}
    best_top1 = 0.0
    
    teacher.to(device)
    student.to(device)
    
    # Initialize learnable temperature module if needed
    learnable_temp_module = None
    if ats_granularity == "learnable":
        if num_classes is None:
            num_classes = len(class_names) if class_names else 10
        
        learnable_temp_module = LearnableTemperatureModule(
            num_classes=num_classes,
            T_base=T_base,
            T_min=T_min,
            T_max=T_max,
            learn_per_class=True
        ).to(device)
        
        # Add learnable parameters to optimizer
        optimizer.add_param_group({'params': learnable_temp_module.parameters()})
        
        print(f"\nLearnable Temperature Module initialized with {num_classes} learnable temperatures")
    
    start_train_timer = timer()
    
    # ========================================================================
    # EPOCH LOOP
    # ========================================================================
    
    for epoch in tqdm(range(epochs)):
        
        # ====================================================================
        # TRAINING STEP WITH ATS
        # ====================================================================
        
        train_metrics = train_step_knowledge_distillation(
            teacher=teacher,
            student=student,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            T_base=T_base,
            T_min=T_min,
            T_max=T_max,
            soft_target_loss_weight=soft_target_loss_weight,
            ce_loss_weight=ce_loss_weight,
            device=device,
            ats_method=ats_method,
            ats_granularity=ats_granularity,
            learnable_temp_module=learnable_temp_module,
            fusion_weights=fusion_weights,
            store_temps=store_temperature_history
        )
        
        # Store temperature statistics
        if store_temperature_history and 'temp_mean' in train_metrics:
            temperature_history.append({
                'epoch': int(epoch),
                'temp_mean': float(train_metrics['temp_mean']),
                'temp_std': float(train_metrics['temp_std']),
                'temp_min': float(train_metrics['temp_min']),
                'temp_max': float(train_metrics['temp_max'])
            })
        elif store_temperature_history and ats_granularity == "learnable":
            if 'per_class' in train_metrics:
                temperature_history.append({
                    'epoch': int(epoch),
                    'per_class_temps': train_metrics['per_class'],
                    'temp_mean': float(train_metrics['mean']),
                    'temp_std': float(train_metrics['std']),
                    'temp_min': float(train_metrics['min']),
                    'temp_max': float(train_metrics['max'])
                })
        
        # ====================================================================
        # VALIDATION STEP
        # ====================================================================
        
        val_metrics = test_step(
            model=student,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        train_losses.append(train_metrics['loss'])
        train_accuracies.append(train_metrics['top1_accuracy'] * 100)
        train_top1_accuracies.append(train_metrics['top1_accuracy'] * 100)
        train_top3_accuracies.append(train_metrics['top3_accuracy'] * 100)
        
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['top1_accuracy'] * 100)
        val_top1_accuracies.append(val_metrics['top1_accuracy'] * 100)
        val_top3_accuracies.append(val_metrics['top3_accuracy'] * 100)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Top-1: {train_metrics['top1_accuracy']*100:.2f}%, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Top-1: {val_metrics['top1_accuracy']*100:.2f}%")
        
        # Print temperature statistics
        if 'temp_mean' in train_metrics:
            print(f"  Temps - Mean: {train_metrics['temp_mean']:.3f}, "
                  f"Std: {train_metrics['temp_std']:.3f}, "
                  f"Range: [{train_metrics['temp_min']:.3f}, {train_metrics['temp_max']:.3f}]")
        
        if val_metrics['loss'] < best_metrics["val_loss"]:
            best_metrics = {"epoch": epoch, "val_loss": val_metrics['loss']}
        
        if val_metrics['top1_accuracy'] > best_top1:
            best_top1 = val_metrics['top1_accuracy']
            save_best_model(student, save_dir, "best_model.pth")
            
            if learnable_temp_module is not None:
                torch.save(learnable_temp_module.state_dict(),
                          os.path.join(save_dir, "best_temp_module.pth"))
            
            print(f"New best student model saved (val_Top-1: {best_top1*100:.2f}%)")
        
        if (epoch + 1) % 20 == 0:
            ckpt_name = f"student_epoch_{epoch+1}.pth"
            save_model(student, checkpoint_dir, ckpt_name)
    
    end_train_timer = timer()
    training_time = end_train_timer - start_train_timer
    print(f"\nTotal training time: {format_time(training_time)}")
    
    # ========================================================================
    # FINAL TESTING
    # ========================================================================
    
    start_test_timer = timer()
    
    test_metrics = test_step(
        model=student,
        dataloader=real_test_dataloader,
        loss_fn=loss_fn,
        device=device
    )
    
    end_test_timer = timer()
    testing_time = end_test_timer - start_test_timer
    
    print(f"\nFinal Test Loss: {test_metrics['loss']:.4f}, "
          f"Test Top-1: {test_metrics['top1_accuracy']*100:.2f}%, "
          f"Test Top-3: {test_metrics['top3_accuracy']*100:.2f}%")
    print(f"Total testing time: {format_time(testing_time)}")
    
    # ========================================================================
    # GENERATE PLOTS AND REPORTS
    # ========================================================================
    
    loss_curve_path, acc_curve_path = plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        save_dir=save_dir
    )
    
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
    
    conf_matrix_path1, y_true1, y_pred1 = generate_confusion_matrix(
        model=student,
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
    
    print(f"Test Classification Report saved to {report_path1}")
    
    conf_matrix_path2, y_true2, y_pred2 = generate_confusion_matrix(
        model=student,
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
    
    print(f"Train Classification Report saved to {report_path2}")
    
    metrics_path = save_training_metrics(
        train_metrics_history=train_metrics_history,
        val_metrics_history=val_metrics_history,
        test_metrics=test_metrics,
        training_time=training_time,
        save_dir=save_dir
    )
    
    detailed_path = save_detailed_metrics(
        train_metrics_history=train_metrics_history,
        val_metrics_history=val_metrics_history,
        test_metrics=test_metrics,
        training_time=training_time,
        save_dir=save_dir
    )
    
    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================
    
    results = {
        "train_losses": [float(x) for x in train_losses],
        "train_accuracies": [float(x) for x in train_accuracies],
        "train_top1_accuracies": [float(x) for x in train_top1_accuracies],
        "train_top3_accuracies": [float(x) for x in train_top3_accuracies],
        "val_losses": [float(x) for x in val_losses],
        "val_accuracies": [float(x) for x in val_accuracies],
        "val_top1_accuracies": [float(x) for x in val_top1_accuracies],
        "val_top3_accuracies": [float(x) for x in val_top3_accuracies],
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_loss": float(best_metrics["val_loss"]),
        "test_loss": float(test_metrics['loss']),
        "test_top1_accuracy": float(test_metrics['top1_accuracy'] * 100),
        "test_top3_accuracy": float(test_metrics['top3_accuracy'] * 100),
        "training_time": float(training_time),
        "testing_time": float(testing_time),
        "ats_method": str(ats_method),
        "ats_granularity": str(ats_granularity),
        "temperature_config": {
            "T_base": float(T_base),
            "T_min": float(T_min),
            "T_max": float(T_max)
        },
        "fusion_weights": {
            "alpha": float(fusion_weights[0]),
            "beta": float(fusion_weights[1]),
            "gamma": float(fusion_weights[2])
        } if ats_granularity == "fusion" else None,
        "temperature_history": temperature_history,
        "plots": {
            "loss_curve": str(loss_curve_path),
            "accuracy_curve": str(acc_curve_path),
            "top1_top3_comparison": str(top1_top3_path),
            "Test_confusion_matrix": str(conf_matrix_path1),
            "Train_confusion_matrix": str(conf_matrix_path2)
        },
        "reports": {
            "Test_classification_report": str(report_path1),
            "Train_classification_report": str(report_path2),
            "training_metrics": str(metrics_path),
            "detailed_metrics": str(detailed_path)
        }
    }
    
    json_path = save_results_to_json(results, save_dir)
    print(f"\nResults saved to {json_path}")
    
    return results

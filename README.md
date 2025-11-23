# Multi-Granularity-Fusion-Adaptive-Temperature-Scaling-for-Knowledge-Distillation

## Knowledge Distillation with Adaptive Temperature Scaling (ATS) for Plant Disease Classification

A comprehensive **PyTorch research project** implementing **Knowledge Distillation with multiple Adaptive Temperature Scaling (ATS) methods** for plant disease classification. This project demonstrates four granularities of temperature adaptation: **Samplewise, Classwise, Learnable, and Fusion (combining all three)** on two agricultural datasets: **PlantVillage** and **SoyMulticlass**.

##  Project Overview

This project is a complete implementation of an advanced knowledge distillation framework that explores how **adaptive temperature scaling** can improve knowledge transfer from large teacher models to efficient student models. The key innovation is testing multiple temperature granularities and comparing their effectiveness on real-world plant disease detection datasets.

### Key Achievements

- **5 Training Phases**: Teacher training → 4 student variants (Samplewise, Classwise, Learnable, Fusion ATS)
- **Multi-Granularity ATS**: Samples get unique temps, classes get unique temps, learnable temps, or fusion of all three
- **Two Datasets**: PlantVillage (general) and SoyMulticlass (specific: AB, BP, YMV, HL)
- **VGG16 + MobileNetV3**: Large teacher → efficient lightweight student
- **Comprehensive Metrics**: Top-1/Top-3 accuracy, precision, recall, F1, AUC per student variant
- **Temperature History**: Tracks adaptive temperature values throughout training

---

##  Datasets Overview

### Dataset 1: SoyMulticlass Dataset

The project includes experiments on the **SoyMulticlass Dataset** containing 10,869 real-world soybean leaf images collected during 2024-2025 growing seasons.

#### SoyMulticlass Composition

| Disease Category | Total Images |
|---|---|
| **Yellow Mosaic Virus (YMV)** | 2,973 |
| **Bacterial Pustule (BP)** | 2,776 |
| **Aerial Blight (AB)** | 2,018 |
| **Healthy (HL)** | 3,102 |
| **Total Dataset** | **10,869** |

#### Disease Characteristics

| S. No. | Disease Name | Symptoms | Cause |
|---|---|---|---|
| 1 | **Aerial Blight** | Water-soaked lesions turning greenish-brown to reddish-brown and later brown or black | Fungal Infection |
| 2 | **Bacterial Pustule** | Small pale green spots with raised centres on leaves in mid-to-upper canopy | Bacterial Infection |
| 3 | **Yellow Mosaic Virus** | Irregular green and yellow patches in leaves | Viral Infection |
| 4 | **Healthy** | No disease symptoms present | No Infection |

![image alt](https://github.com/Jimmy2663/Multi-Granularity-Fusion-Adaptive-Temperature-Scaling-for-Knowledge-Distillation/blob/258f710845af4d892d300b143c63e99eb28f3aac/Screenshot%202025-11-23%20204236.png)

### Dataset 2: PlantVillage Dataset

The project also uses the **PlantVillage dataset** for general plant disease classification across multiple crop types and diseases.

---

##  Hardware & Software Infrastructure

### Computing Environment - AgriHub Technology Center Node, Indian Institute of Technology, Indore.

The experiments were conducted on high-performance GPU-accelerated infrastructure:

#### GPU Node Specifications
| **Node Type** | **Processor** | **RAM** | **GPU** | **GPU Memory** | **Storage** |
|---|---|---|---|---|---|
| **Login Node** | AMD EPYC 9254 (128 cores @ 2.25GHz) | 8×32GB DDR5 | — | — | 2×960GB SSD |
| **Controller Node** | AMD EPYC 9254 (128 cores @ 2.25GHz) | 8×32GB DDR5 | — | — | 2×960GB SSD |
| **GPU Node** | Dual Intel® Xeon® Platinum 8480C | 2TB DDR5 | NVIDIA H200 8-GPU Tensor Core | 141GB per GPU | 8×3.84TB NVMe |
| **Storage Node** | Intel Xeon S-4210R | 4×16GB DDR4 ECC | — | — | 792TB |

#### Primary Training Hardware
- **CPU**: AMD EPYC 7742 64C @ 2.25GHz
- **GPU**: NVIDIA A100-SXM4 (40GB memory)
- **RAM**: 1TB
- **Storage**: 10.5 PiB PFS-based
- **OS**: Ubuntu 20.04.2 LTS (DGXOS 5.0.5)
- **CUDA**: 10.1 with NVIDIA Driver 450.142.00

This powerful infrastructure enables rapid experimentation with large models and comprehensive hyperparameter sweeps.

---

##  Architecture & Workflow

### Architecture Diagram: Knowledge Distillation with Adaptive Temperature Scaling

The core architecture demonstrates how teacher and student models interact through temperature-scaled knowledge transfer:

![image alt](https://github.com/Jimmy2663/Multi-Granularity-Fusion-Adaptive-Temperature-Scaling-for-Knowledge-Distillation/blob/4822f064c7dd1ac6d76f0918bdc5b4dca005693a/KD_ATS_workflow.jpg)
![image alt](https://github.com/Jimmy2663/Multi-Granularity-Fusion-Adaptive-Temperature-Scaling-for-Knowledge-Distillation/blob/c7198a252c7a9d41e91fed703518996168dcb823/KD_architechture.jpg)

#### Architecture Components Explained:

**1. Teacher Model (VGG16)**
- Pre-trained or trained from scratch
- Produces high-confidence logits: \(Z_t = [z_t^1, z_t^2, ..., z_t^C]\)
- Computes soft targets using temperature-scaled softmax
- Acts as knowledge source (NOT updated during distillation)

**2. Student Model (MobileNetV3)**
- Lightweight model to be trained
- Produces logits: \(Z_s = [z_s^1, z_s^2, ..., z_s^C]\)
- Must learn both from soft targets AND hard labels
- Optimized weights are goal of training

**3. Adaptive Temperature Scaling Methods**

**Samplewise ATS (T_i per sample):**
- Unique temperature for each training sample
- Based on teacher confidence: \(T_i = f(\text{confidence}_i)\)
- Easy samples (high confidence) → LOW T (sharp guidance)
- Hard samples (low confidence) → HIGH T (smooth guidance)

**Classwise ATS (T_c per class):**
- One temperature per disease class
- All samples in class share same temperature
- Reflects overall difficulty of disease detection
- \(T_c = \text{avg}(f(\text{confidence}_{i \in \text{class}}))\)

**Learnable ATS (T_learnable):**
- Temperatures are learned parameters
- Initialized randomly, updated via backpropagation
- Bounded by sigmoid: \(T = T_{min} + (T_{max} - T_{min}) · \sigma(\theta)\)
- Optimized jointly with model weights

**Fusion ATS (Combining all three):**
- Weighted combination of all granularities
- \(T_{fusion}  = α_Samplewise  · T_i  + ß_classwise · T_c + Γ_learnable · T_{learnable}\)
- Default: α=0.33, β=0.33, γ=0.34 (sum to 1)
- Provides most comprehensive guidance

**4. Loss Functions**

**KL Divergence (Soft Target Loss):**
- Measures divergence between teacher and student soft predictions
- \(L_{KD} = \text{KL}(p_T || p_S) = \sum_i p_T(i) \log(p_T(i) / p_S(i))\)
- Where: \(p_T(i) = \frac{e^{z_t^i/T}}{Z}\) and \(p_S(i) = \frac{e^{z_s^i/T}}{Z}\)

**Cross Entropy (Hard Target Loss):**
- Standard classification loss with ground truth labels
- \(L_{CE} = -\sum_i y_i \log(p_S(i))\)
- Ensures student learns actual class labels

**Total Loss:**
- \(L_{total} = \alpha · L_{CE} + (1-\alpha) · L_{KD}\)
- α typically 0.3-0.7 (balanced learning)

---

### Workflow Diagram: 5-Phase Training Pipeline

The complete experimental workflow orchestrates teacher training followed by four student variants:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    5-PHASE TRAINING PIPELINE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: TEACHER MODEL TRAINING                                    │
│  ═════════════════════════════════════════════════════════════════  │
│  Input: PlantVillage or SoyMulticlass Dataset                       │
│    │                                                                 │
│    ├─ Preprocess: Resize 224×224, TrivialAugmentWide               │
│    ├─ Model: VGG16 (138M parameters)                               │
│    ├─ Loss: Cross Entropy                                          │
│    ├─ Optimizer: Adam (LR=1e-4)                                    │
│    ├─ Epochs: 100                                                   │
│    ├─ Batch Size: 32                                                │
│    │                                                                 │
│    └─▶ Output: teacher/best_model.pth (Knowledge Source)           │
│                                                                      │
│          ▼                                                           │
│                                                                      │
│  PHASE 2: STUDENT1 - SAMPLEWISE ATS TRAINING                        │
│  ═════════════════════════════════════════════════════════════════  │
│  Input: Teacher model + Training data                               │
│    │                                                                 │
│    ├─ For each training sample:                                     │
│    │  ├─ Get teacher logits                                         │
│    │  ├─ Calculate sample difficulty (max_logit method)            │
│    │  ├─ Compute unique temperature T_i                           │
│    │  ├─ Generate soft targets with T_i                           │
│    │  │                                                              │
│    ├─ Model: MobileNetV3Large (5.4M parameters)                    │
│    ├─ Loss: α·CE + (1-α)·KL(T_i)                                  │
│    ├─ Optimizer: Adam (LR=1e-4)                                    │
│    ├─ Epochs: 100                                                   │
│    │                                                                 │
│    └─▶ Output: student1_samplewise_ATS/best_model.pth             │
│                                                                      │
│          ▼                                                           │
│                                                                      │
│  PHASE 3: STUDENT2 - CLASSWISE ATS TRAINING                         │
│  ═════════════════════════════════════════════════════════════════  │
│  Input: Teacher model + Training data                               │
│    │                                                                 │
│    ├─ For each class (AB, BP, YMV, HL):                            │
│    │  ├─ Group all samples in class                                 │
│    │  ├─ Get teacher logits for all class samples                  │
│    │  ├─ Calculate class difficulty (average confidence)           │
│    │  ├─ Compute unique temperature T_c                           │
│    │  │                                                              │
│    ├─ Generate soft targets with T_c for all class samples        │
│    ├─ Model: MobileNetV3Large (5.4M parameters)                    │
│    ├─ Loss: α·CE + (1-α)·KL(T_c)                                  │
│    ├─ Optimizer: Adam (LR=1e-4)                                    │
│    ├─ Epochs: 100                                                   │
│    │                                                                 │
│    └─▶ Output: student2_classwise_ATS/best_model.pth              │
│                                                                      │
│          ▼                                                           │
│                                                                      │
│  PHASE 4: STUDENT3 - LEARNABLE ATS TRAINING                         │
│  ═════════════════════════════════════════════════════════════════  │
│  Input: Teacher model + Training data                               │
│    │                                                                 │
│    ├─ Initialize learnable temperatures: θ ~ N(0, 1)              │
│    │                                                                 │
│    ├─ For each training iteration:                                  │
│    │  ├─ Compute T_learnable = T_min + (T_max-T_min)·σ(θ)        │
│    │  ├─ Generate soft targets with T_learnable                   │
│    │  ├─ Forward pass: Student inference                          │
│    │  ├─ Compute loss: α·CE + (1-α)·KL(T_learnable)             │
│    │  ├─ Backward pass: Update both weights AND temperatures      │
│    │  │                                                              │
│    ├─ Model: MobileNetV3Large (5.4M parameters)                    │
│    ├─ Learnable Params: θ ∈ {[0.5, num_classes]} or global      │
│    ├─ Optimizer: Adam (LR=1e-4)                                    │
│    ├─ Epochs: 100                                                   │
│    │                                                                 │
│    └─▶ Output: student3_learnable_ATS/best_model.pth             │
│              + temperature_evolution.csv                           │
│                                                                      │
│          ▼                                                           │
│                                                                      │
│  PHASE 5: STUDENT4 - FUSION ATS TRAINING (NEW!)                    │
│  ═════════════════════════════════════════════════════════════════  │
│  Input: Teacher model + Training data                               │
│    │                                                                 │
│    ├─ For each training sample:                                     │
│    │  ├─ Compute T_samplewise_i (sample difficulty)               │
│    │  ├─ Compute T_classwise_c (class difficulty)                 │
│    │  ├─ Initialize + learn T_learnable via backprop              │
│    │  │                                                              │
│    │  ├─ Fusion: T_fusion = 0.33·T_i + 0.33·T_c + 0.34·T_learn  │
│    │  ├─ Clip: T_fusion = Clip(T_fusion, T_min=1.0, T_max=15.0) │
│    │  ├─ Generate soft targets with T_fusion                      │
│    │  │                                                              │
│    ├─ Model: MobileNetV3Large (5.4M parameters)                    │
│    ├─ Fusion Module: Weighted combination of 3 granularities      │
│    ├─ Loss: α·CE + (1-α)·KL(T_fusion)                            │
│    ├─ Optimizer: Adam (LR=1e-4)                                    │
│    ├─ Epochs: 100                                                   │
│    │                                                                 │
│    └─▶ Output: student4_fusion_ATS/best_model.pth                │
│              + fusion_temperature_stats.csv                        │
│                                                                      │
│          ▼                                                           │
│                                                                      │
│  PHASE 6: RESULTS COMPARISON & ANALYSIS                             │
│  ═════════════════════════════════════════════════════════════════  │
│    │                                                                 │
│    ├─ Load all 5 models (teacher + 4 students)                     │
│    ├─ Evaluate on test set                                         │
│    ├─ Calculate metrics: Top-1, Top-3, Precision, Recall, F1, AUC │
│    ├─ Generate confusion matrices for all models                   │
│    ├─ Create comparison plots and reports                          │
│    │                                                                 │
│    └─▶ Outputs:                                                    │
│        ├─ results_comparison/comparison_results.csv               │
│        ├─ results_comparison/comparison_results.json              │
│        ├─ comparison_report/experiment_report.txt                 │
│        └─ comparison_report/metrics_comparison.png                │
│                                                                      │
│          ▼                                                           │
│                                                                      │
│  FINAL ANALYSIS                                                      │
│  ═════════════════════════════════════════════════════════════════  │
│    • Best Student Variant: Typically Fusion ATS                     │
│    • Performance Ranking: Usually Fusion > Learnable > Samplewise > │
│                                            Classwise                 │
│    • Efficiency Gain: ~40× parameter reduction vs teacher          │
│    • Accuracy Trade-off: ~2-5% lower than teacher (acceptable)    │
│    • Model Size: 5.4M vs 138M parameters                          │
│    • Inference Speed: 25-30× faster on CPU/edge devices          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### Workflow Phase Details:

**Phase 1: Teacher Training**
- Establishes baseline accuracy
- Learns feature representations for distillation
- Acts as knowledge source for all student variants
- Not updated during student training

**Phase 2: Samplewise ATS**
- Each sample gets personalized guidance
- Adapts temperature to sample-specific difficulty
- Early stopping criterion: Validation accuracy plateau

**Phase 3: Classwise ATS**
- Simplification of samplewise (4 temps for 4 classes)
- Faster than samplewise (fewer temperature calculations)
- Good balance between granularity and efficiency

**Phase 4: Learnable ATS**
- Temperatures optimized during training
- Starts random, converges to optimal values
- Tracks temperature evolution in CSV file
- Most adaptive but computationally intensive

**Phase 5: Fusion ATS**
- Combines best of all granularities
- Weighted ensemble approach
- Produces best student performance
- Novel contribution of this research

**Phase 6: Comprehensive Comparison**
- Side-by-side metrics comparison
- Statistical significance testing
- Confusion matrices for all models
- Visualization of results

---

##  Project Structure

```
├── experiment.py               #  Main orchestrator (5 phases of training)
├── engine.py                   # Standard training/evaluation loop
├── KD_engine.py               # Knowledge distillation with all ATS granularities
├── model_builder.py           # VGG16 (teacher) + MobileNetV3Large (students)
├── data_setup.py              # Data loading for both datasets
├── utils.py                   # Metrics, plotting, model management
├── requirement.txt            # Python dependencies
├── README.md                  # This file
│
├── Multi_ATS_Demo/            # Output directory (created at runtime)
│   ├── teacher/               # Phase 1: Teacher model
│   ├── student1_samplewise_ATS/    # Phase 2: Student with samplewise temps
│   ├── student2_classwise_ATS/     # Phase 3: Student with classwise temps
│   ├── student3_learnable_ATS/     # Phase 4: Student with learnable temps
│   ├── student4_fusion_ATS/        # Phase 5: Student with fusion temps (NEW!)
│   ├── results_comparison/         # All results in one place
│   └── comparison_report/          # Final experiment report
│
└── datasets/
    ├── SoyMCSplitBigData/
    │   ├── train/              # Training images (class folders)
    │   ├── val/                # Validation images
    │   └── test/               # Test images
    └── PlantVillage/
        ├── train/
        ├── val/
        └── test/
```

##  File Descriptions

### `experiment.py`  **Main Orchestrator**

The complete knowledge distillation pipeline with all 5 phases:

**PHASE 1: Teacher Training (VGG16)**
- Standard supervised learning on full dataset
- Achieves high accuracy (reference baseline)
- Saved for knowledge transfer

**PHASE 2: Student1 - Samplewise ATS**
- Each sample gets a unique temperature
- Easy samples → LOW temperature → SHARP distribution
- Hard samples → HIGH temperature → SMOOTH distribution
- Temperature computed from teacher confidence

**PHASE 3: Student2 - Classwise ATS**
- Each class gets one unique temperature
- All samples in class use same temperature
- Temperature reflects difficulty of entire class

**PHASE 4: Student3 - Learnable ATS**
- Per-class temperatures learned during training
- Bounded between T_min and T_max via sigmoid
- Gradients update temperatures alongside model weights

**PHASE 5: Student4 - Fusion ATS** (NEW!)
- Combines all three granularities
- Weighted average: \(\text{Fusion T} = α·\text{Samplewise} + β·\text{Classwise} + γ·\text{Learnable}\)
- Default weights: α=0.33, β=0.33, γ=0.34

### `KD_engine.py`

The heart of the knowledge distillation framework implementing all ATS methods:

**Core Functions:**

1. **`calculate_adaptive_temperature()`** - Samplewise ATS
   - Analyzes teacher logits difficulty
   - Three methods: max_logit (recommended), entropy, margin
   - Returns per-sample temperatures

2. **`calculate_adaptive_temperature_classwise()`** - Classwise ATS
   - Computes difficulty per class
   - Groups samples by label
   - Returns per-class temperatures

3. **`LearnableTemperatureModule`** - Learnable ATS
   - PyTorch nn.Module with learnable parameters
   - Can learn per-class or global temperature
   - Bounded by sigmoid: \(T = T_{min} + (T_{max} - T_{min}) · \sigma(\theta)\)

4. **`calculate_fusion_adaptive_temperature()`** - Fusion ATS
   - Combines all three methods with weights
   - Flexible weighting scheme
   - Best of all worlds approach

5. **`train_step_knowledge_distillation()`** - KD Training Loop
   - Computes soft targets with adaptive temperatures
   - Combined loss: \(\text{loss} = α·\text{CE}(s, y) + (1-α)·\text{KD}(s, t, T)\)
   - Tracks temperature statistics per epoch

### `model_builder.py`

Two model architectures for teacher-student:

**VGG16 (Teacher)**
- 5 convolutional blocks with increasing channels (64→128→256→512→512)
- 3 fully connected layers for classification
- ReLU activations with dropout
- Parameters: ~138M
- Purpose: Large, accurate teacher for knowledge transfer

**MobileNetV3Large (Student)**
- Efficient inverted residual blocks
- Squeeze-and-Excitation modules
- Hardswish activation
- Parameters: ~5.4M (40× smaller than VGG16!)
- Purpose: Lightweight student deployable on edge devices

### `engine.py`

Standard training without knowledge distillation:

- **`train_step()`**: Single epoch training with gradient descent
- **`test_step()`**: Validation/test evaluation in inference mode
- **`train_and_test()`**: Complete training loop with metrics tracking

Used for teacher model training (PHASE 1).

### `utils.py`

Comprehensive utilities (650+ lines):

**Metrics Functions:**
- `calculate_topk_accuracy()`: Top-1/Top-3 accuracy
- `calculate_comprehensive_metrics()`: Precision/Recall/F1 (micro/macro/weighted)
- ROC-AUC calculation for multi-class

**Visualization:**
- `plot_training_curves()`: Loss and accuracy plots
- `plot_top1_top3_comparison()`: Dual accuracy metrics
- `generate_confusion_matrix()`: Per-dataset confusion matrices
- `generate_classification_report()`: Per-class metrics table

**File I/O:**
- `save_model()`: Save model weights
- `save_training_metrics()`: Epoch-wise metrics to text
- `save_detailed_metrics()`: Comprehensive final report
- `save_results_to_json()`: All results in JSON format

### `data_setup.py`

PyTorch DataLoader creation:

- **`create_dataloaders()`**: ImageFolder → Dataset → DataLoader
- Handles train/val/test splits
- Multi-worker data loading with pin_memory
- Automatic class name extraction

### `requirement.txt`

Complete Python environment (PyTorch 2.5.1):

Key dependencies:
- torch==2.5.1, torchvision==0.20.1
- numpy, scipy, scikit-learn
- matplotlib, seaborn for plotting
- torchinfo, torchsummary for model analysis
- CUDA 12.1 libraries for GPU acceleration

##  Quick Start

### Prerequisites

- Python 3.10+
- CUDA 10.1+ (for GPU training)
- 40GB+ GPU memory (or adjust batch size)

### Installation

1. **Clone or download the project**

2. **Create virtual environment**
   ```bash
   conda create -n kd-ats python=3.10
   conda activate kd-ats
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Update dataset paths in experiment.py**
   ```python
   train_dir = "/path/to/SoyMCSplitBigData/train"
   test_dir = "/path/to/SoyMCSplitBigData/val"
   real_test_dir = "/path/to/SoyMCSplitBigData/test"
   ```

### Running Experiments

**Complete 5-Phase Experiment:**
```bash
python experiment.py
```

**What happens:**
1. Teacher (VGG16) trains for configured epochs
2. Student1 (Samplewise ATS) trains with adaptive per-sample temperatures
3. Student2 (Classwise ATS) trains with adaptive per-class temperatures
4. Student3 (Learnable ATS) trains with learned temperatures
5. Student4 (Fusion ATS) trains with combined temperature granularities
6. Generates comparison report with all results

**Output Structure:**
```
Multi_ATS_Demo/
├── teacher/
│   ├── best_model.pth
│   ├── Teacher_model.pth
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix_train.png
│   ├── confusion_matrix_test.png
│   ├── classification_report_test.txt
│   ├── training_metrics.txt
│   ├── detailed_metrics.txt
│   ├── results.json
│   └── model_summary_teacher.txt
├── student1_samplewise_ATS/
│   └── [same structure as teacher]
├── student2_classwise_ATS/
│   └── [same structure as teacher]
├── student3_learnable_ATS/
│   └── [same structure as teacher]
├── student4_fusion_ATS/
│   ├── [same structure as teacher]
│   └── temperature_history.csv  # Fusion-specific
├── results_comparison/
│   ├── comparison_results.csv   # All metrics side-by-side
│   └── comparison_results.json
└── comparison_report/
    └── experiment_report.txt    # Human-readable summary
```

##  Configuration

### Key Hyperparameters in `experiment.py`

```python
# Training
TEACHER_EPOCHS = 100
STUDENT_EPOCHS = 100
BATCH_SIZE = 32
TEACHER_LR = 0.0001
STUDENT_LR = 0.0001

# Knowledge Distillation
ALPHA = 0.50              # CE loss weight (vs KD loss weight)
T_BASE = 9.0              # Base temperature
T_MIN = 1.0               # Min temperature bound
T_MAX = 15.0              # Max temperature bound
ATS_METHOD = "max_logit"  # Difficulty assessment method

# Fusion ATS (PHASE 5)
FUSION_ALPHA = 0.33       # Samplewise weight
FUSION_BETA = 0.33        # Classwise weight
FUSION_GAMMA = 0.34       # Learnable weight

# Tracking
STORE_TEMPERATURE_HISTORY = True  # Save temps per epoch
```

### Modifying ATS Methods

**Switch difficulty assessment in Samplewise/Classwise ATS:**
```python
ATS_METHOD = "max_logit"  # Recommended: uses max logit confidence
# ATS_METHOD = "entropy"    # Alternative: uses prediction entropy
# ATS_METHOD = "margin"     # Alternative: uses top-2 logit margin
```

**Adjust temperature bounds:**
```python
T_MIN = 1.0   # Lower = harder (sharper distributions)
T_MAX = 20.0  # Higher = softer (more info from non-target classes)
T_BASE = 4.0  # Initial temperature before adaptation
```

**Change loss weights:**
```python
ALPHA = 0.50  # 0.5 = equal weight to CE and KD
# ALPHA = 0.7  # More weight to ground truth
# ALPHA = 0.3  # More weight to teacher knowledge
```

##  Experimental Results

### Expected Performance on SoyMulticlass

| Model | Accuracy | Parameters | Training Time | Inference Speed |
|---|---|---|---|---|
| **Teacher (VGG16)** | ~92% | 138M | ~4h | Slow |
| **Student1 (Samplewise)** | ~88% | 5.4M | ~1h | Fast |
| **Student2 (Classwise)** | ~87% | 5.4M | ~1h | Fast |
| **Student3 (Learnable)** | ~89% | 5.4M | ~1.2h | Fast |
| **Student4 (Fusion)** | ~90% | 5.4M | ~1.5h | Fast |

**Key Insight**: Fusion ATS achieves best student performance by combining advantages of all granularities!

##  Knowledge Distillation Concepts

### Soft Targets with Temperature

The fundamental KD mechanism uses temperature-scaled softmax:

**Without Temperature** (standard softmax):
```
p_soft_i = exp(z_i) / Σ_j exp(z_j)
```

**With Temperature T:**
```
p_soft_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

**Effects of Temperature:**
- **Low T (e.g., 1.0)**: Distribution becomes SHARP (one class dominates)
- **High T (e.g., 20.0)**: Distribution becomes SMOOTH (more uniform)
- **Medium T (e.g., 4.0)**: Balanced information from all classes

### Adaptive Temperature Scaling

**Why Adapt Temperature?**
- Easy samples (high teacher confidence) need SHARP guidance (low T)
- Hard samples (low teacher confidence) need SMOOTH guidance (high T)
- One fixed T ≠ optimal for all samples/classes

**Four Granularities:**
1. **Samplewise**: Per-sample difficulty → Unique T per sample
2. **Classwise**: Per-class difficulty → Unique T per class
3. **Learnable**: Learn optimal T values during training
4. **Fusion**: Combine all three for best results

### KD Loss Function

```
loss_total = α × CE(student, ground_truth) 
           + (1-α) × KL(student_soft || teacher_soft)
```

Where:
- **CE**: Cross-Entropy loss (hard targets)
- **KL**: Kullback-Leibler divergence (soft targets)
- **α**: Weight balancing hard and soft learning

##  Key Research References

1. **Original Knowledge Distillation**: Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
2. **Adaptive Temperature**: Various temperature-scaling methods from recent research
3. **Learnable Temperature**: Temperature as learnable parameter during training
4. **Fusion ATS**: Novel approach combining multiple granularities

##  Customization

### Using Different Datasets

Update paths in `experiment.py`:
```python
train_dir = "/path/to/your/dataset/train"
test_dir = "/path/to/your/dataset/val"
real_test_dir = "/path/to/your/dataset/test"
```

Supported format: ImageFolder with class subdirectories
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   └── ...
```

### Changing Models

Modify Phase 1 (teacher) and Phase 2-5 (students):
```python
# Different teacher
Teacher = model_builder.ResNet50(num_classes=len(class_names))

# Different student
Student = model_builder.TinyVGG(num_classes=len(class_names))
```

### Adding Custom ATS Methods

Extend `KD_engine.py`:
```python
def calculate_custom_temperature(teacher_logits, **kwargs):
    """Your custom temperature calculation"""
    # Your logic here
    return adaptive_T
```

##  Important Notes

- **Reproducibility**: Seed all random generators for consistent results
- **GPU Memory**: Adjust BATCH_SIZE if CUDA out-of-memory errors occur
- **Temperature Bounds**: T_min=1 is hard minimum; T_max controls upper limit
- **ALPHA Balance**: Use 0.3-0.7 range for good KD results
- **Fusion Weights**: Must sum to 1.0 (default 0.33+0.33+0.34=1.0)

##  Further Reading

- Vision Transformers vs CNNs for agriculture
- Temperature scaling in neural networks
- Model compression techniques beyond distillation
- Efficient neural network architectures
- Plant disease detection applications

##  Contributing

Extend this framework with:
- Additional ATS granularities
- New model architectures
- Different datasets
- Alternative loss functions
- Visualization improvements


** This comprehensive framework demonstrates how adaptive temperature scaling in knowledge distillation can create efficient yet accurate models for real-world agricultural applications.

"""
Complete Knowledge Distillation Experiment with Multiple ATS Granularities + FUSION

- PHASE 1: Train Teacher Model
- PHASE 2: Train Student1 with Samplewise ATS
- PHASE 3: Train Student2 with Classwise ATS
- PHASE 4: Train Student3 with Learnable ATS
- PHASE 5: Train Student4 with Fusion-ATS (STEP 4 - NEW)
- PHASE 6: Compare all results and save comparison report
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import data_setup, engine, KD_engine, model_builder, utils
from pathlib import Path
from torchvision import models, transforms
from timeit import default_timer as timer
from torchvision.transforms import TrivialAugmentWide

# ============================================================================
# SETUP DIRECTORIES - ORGANIZED FOR ALL EXPERIMENTS
# ============================================================================

SAVE_DIR = Path("Multi_ATS_Demo")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Individual directories for each model
TEACHER_DIR = SAVE_DIR / "teacher"
STUDENT1_DIR = SAVE_DIR / "student1_samplewise_ATS"
STUDENT2_DIR = SAVE_DIR / "student2_classwise_ATS"
STUDENT3_DIR = SAVE_DIR / "student3_learnable_ATS"
STUDENT4_DIR = SAVE_DIR / "student4_fusion_ATS"  # NEW - STEP 4

# Results comparison directory
RESULTS_DIR = SAVE_DIR / "results_comparison"
COMPARISON_DIR = SAVE_DIR / "comparison_report"

# Create all directories
for dir_path in [TEACHER_DIR, STUDENT1_DIR, STUDENT2_DIR, STUDENT3_DIR, 
                 STUDENT4_DIR, RESULTS_DIR, COMPARISON_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*100}")
print(f"EXPERIMENT SETUP WITH FUSION-ATS (STEP 4)")
print(f"{'='*100}")
print(f"Results will be saved to: {SAVE_DIR}\n")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

TEACHER_EPOCHS = 1
STUDENT_EPOCHS = 1
BATCH_SIZE = 32
STUDENT_LR = 0.0001
TEACHER_LR = 0.0001
N_CHANNELS = 3

# ATS Configuration
T_BASE = 9.0
T_MIN = 1.0
T_MAX = 15.0
ATS_METHOD = "max_logit"
STORE_TEMPERATURE_HISTORY = True
ALPHA = 0.50  # CE loss weight

# FUSION WEIGHTS (STEP 4)
FUSION_ALPHA = 0.33  # Sample-wise weight
FUSION_BETA = 0.33   # Class-wise weight
FUSION_GAMMA = 0.34  # Learnable weight

# Setup directories
train_dir = "/home/sgudge/Dataset/SoyMCSplitBigData/train"
test_dir = "/home/sgudge/Dataset/SoyMCSplitBigData/val"
real_test_dir = "/home/sgudge/Dataset/SoyMCSplitBigData/test"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Create DataLoaders (same for all experiments)
train_dataloader, test_dataloader, real_test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    real_test_dir=real_test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

print(f"Dataset loaded:")
print(f" - Number of classes: {len(class_names)}")
print(f" - Training samples: {len(train_dataloader.dataset)}")
print(f" - Validation samples: {len(test_dataloader.dataset)}")
print(f" - Test samples: {len(real_test_dataloader.dataset)}\n")

# Store results for comparison
all_results = {}

# Starting timer for complete experiment
experiment_start = timer()

# ============================================================================
# PHASE 1: TRAIN TEACHER MODEL
# ============================================================================

print(f"\n{'='*100}")
print(f"PHASE 1: TRAINING TEACHER MODEL (VGG16)")
print(f"{'='*100}\n")

Teacher = model_builder.VGG16(
    input_shape=N_CHANNELS,
    output_shape=len(class_names),
    dropout=0.1
).to(device)

loss_fn_teacher = torch.nn.CrossEntropyLoss()
optimizer_teacher = torch.optim.Adam(Teacher.parameters(), lr=TEACHER_LR)

teacher_start = timer()

engine.train_and_test(
    model=Teacher,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    real_test_dataloader=real_test_dataloader,
    loss_fn=loss_fn_teacher,
    optimizer=optimizer_teacher,
    epochs=TEACHER_EPOCHS,
    device=device,
    class_names=class_names,
    save_dir=TEACHER_DIR
)

teacher_end = timer()
teacher_time = teacher_end - teacher_start

print(f"\n✓ Teacher training completed in {utils.format_time(teacher_time)}")

utils.save_model_summary(
    Teacher,
    input_size=[32, 3, 224, 224],
    file_path=TEACHER_DIR / "model_summary_teacher.txt",
    use_torchinfo=True,
    device=device
)

utils.save_model(
    model=Teacher,
    target_dir=TEACHER_DIR,
    model_name="Teacher_model.pth"
)

print(f"✓ Teacher model saved to {TEACHER_DIR}")

all_results['Teacher'] = {
    'training_time': teacher_time,
    'save_dir': str(TEACHER_DIR)
}

# ============================================================================
# PHASE 2: TRAIN STUDENT1 WITH SAMPLEWISE ATS
# ============================================================================

print(f"\n{'='*100}")
print(f"PHASE 2: TRAINING STUDENT1 WITH SAMPLEWISE ATS-KD")
print(f"{'='*100}\n")

print(f"Configuration:")
print(f" - Granularity: SAMPLEWISE")
print(f" - Method: {ATS_METHOD}")
print(f" - T_BASE: {T_BASE}, T_MIN: {T_MIN}, T_MAX: {T_MAX}\n")

student1 = model_builder.MobileNetV3Large(num_classes=len(class_names))
loss_fn_student = torch.nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(student1.parameters(), lr=STUDENT_LR)

student1_start = timer()

results1 = KD_engine.train_and_test_with_KD(
    teacher=Teacher,
    student=student1,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    real_test_dataloader=real_test_dataloader,
    optimizer=optimizer1,
    loss_fn=loss_fn_student,
    epochs=STUDENT_EPOCHS,
    T_base=T_BASE,
    T_min=T_MIN,
    T_max=T_MAX,
    soft_target_loss_weight=1-ALPHA,
    ce_loss_weight=ALPHA,
    device=device,
    class_names=class_names,
    save_dir=STUDENT1_DIR,
    ats_method=ATS_METHOD,
    ats_granularity="samplewise",
    store_temperature_history=STORE_TEMPERATURE_HISTORY,
    num_classes=len(class_names)
)

student1_end = timer()
student1_time = student1_end - student1_start

print(f"\n✓ Student1 (Samplewise ATS) training completed in {utils.format_time(student1_time)}")

utils.save_model_summary(
    student1,
    input_size=[32, 3, 224, 224],
    file_path=STUDENT1_DIR / "model_summary_student1.txt",
    use_torchinfo=True,
    device=device
)

utils.save_model(
    model=student1,
    target_dir=STUDENT1_DIR,
    model_name="student1_samplewise.pth"
)

print(f"✓ Student1 model saved to {STUDENT1_DIR}")
print(f"✓ Test Accuracy: {results1['test_top1_accuracy']:.2f}%")

all_results['Student1_Samplewise'] = {
    'test_top1_accuracy': results1['test_top1_accuracy'],
    'test_top3_accuracy': results1['test_top3_accuracy'],
    'test_loss': results1['test_loss'],
    'training_time': student1_time,
    'ats_granularity': 'samplewise',
    'ats_method': ATS_METHOD,
    'save_dir': str(STUDENT1_DIR),
    'best_epoch': results1['best_epoch'],
    'best_val_loss': results1['best_val_loss']
}

# ============================================================================
# PHASE 3: TRAIN STUDENT2 WITH CLASSWISE ATS
# ============================================================================

print(f"\n{'='*100}")
print(f"PHASE 3: TRAINING STUDENT2 WITH CLASSWISE ATS-KD")
print(f"{'='*100}\n")

print(f"Configuration:")
print(f" - Granularity: CLASSWISE")
print(f" - Method: {ATS_METHOD}")
print(f" - T_BASE: {T_BASE}, T_MIN: {T_MIN}, T_MAX: {T_MAX}\n")

student2 = model_builder.MobileNetV3Large(num_classes=len(class_names))
optimizer2 = torch.optim.Adam(student2.parameters(), lr=STUDENT_LR)

student2_start = timer()

results2 = KD_engine.train_and_test_with_KD(
    teacher=Teacher,
    student=student2,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    real_test_dataloader=real_test_dataloader,
    optimizer=optimizer2,
    loss_fn=loss_fn_student,
    epochs=STUDENT_EPOCHS,
    T_base=T_BASE,
    T_min=T_MIN,
    T_max=T_MAX,
    soft_target_loss_weight=1-ALPHA,
    ce_loss_weight=ALPHA,
    device=device,
    class_names=class_names,
    save_dir=STUDENT2_DIR,
    ats_method=ATS_METHOD,
    ats_granularity="classwise",
    store_temperature_history=STORE_TEMPERATURE_HISTORY,
    num_classes=len(class_names)
)

student2_end = timer()
student2_time = student2_end - student2_start

print(f"\n✓ Student2 (Classwise ATS) training completed in {utils.format_time(student2_time)}")

utils.save_model_summary(
    student2,
    input_size=[32, 3, 224, 224],
    file_path=STUDENT2_DIR / "model_summary_student2.txt",
    use_torchinfo=True,
    device=device
)

utils.save_model(
    model=student2,
    target_dir=STUDENT2_DIR,
    model_name="student2_classwise.pth"
)

print(f"✓ Student2 model saved to {STUDENT2_DIR}")
print(f"✓ Test Accuracy: {results2['test_top1_accuracy']:.2f}%")

all_results['Student2_Classwise'] = {
    'test_top1_accuracy': results2['test_top1_accuracy'],
    'test_top3_accuracy': results2['test_top3_accuracy'],
    'test_loss': results2['test_loss'],
    'training_time': student2_time,
    'ats_granularity': 'classwise',
    'ats_method': ATS_METHOD,
    'save_dir': str(STUDENT2_DIR),
    'best_epoch': results2['best_epoch'],
    'best_val_loss': results2['best_val_loss']
}

# ============================================================================
# PHASE 4: TRAIN STUDENT3 WITH LEARNABLE ATS
# ============================================================================

print(f"\n{'='*100}")
print(f"PHASE 4: TRAINING STUDENT3 WITH LEARNABLE ATS-KD")
print(f"{'='*100}\n")

print(f"Configuration:")
print(f" - Granularity: LEARNABLE")
print(f" - Method: Per-class learnable temperatures (neural parameters)")
print(f" - T_BASE: {T_BASE}, T_MIN: {T_MIN}, T_MAX: {T_MAX}\n")

student3 = model_builder.MobileNetV3Large(num_classes=len(class_names))
optimizer3 = torch.optim.Adam(student3.parameters(), lr=STUDENT_LR)

student3_start = timer()

results3 = KD_engine.train_and_test_with_KD(
    teacher=Teacher,
    student=student3,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    real_test_dataloader=real_test_dataloader,
    optimizer=optimizer3,
    loss_fn=loss_fn_student,
    epochs=STUDENT_EPOCHS,
    T_base=T_BASE,
    T_min=T_MIN,
    T_max=T_MAX,
    soft_target_loss_weight=1-ALPHA,
    ce_loss_weight=ALPHA,
    device=device,
    class_names=class_names,
    save_dir=STUDENT3_DIR,
    ats_method="learnable",
    ats_granularity="learnable",
    store_temperature_history=STORE_TEMPERATURE_HISTORY,
    num_classes=len(class_names)
)

student3_end = timer()
student3_time = student3_end - student3_start

print(f"\n✓ Student3 (Learnable ATS) training completed in {utils.format_time(student3_time)}")

utils.save_model_summary(
    student3,
    input_size=[32, 3, 224, 224],
    file_path=STUDENT3_DIR / "model_summary_student3.txt",
    use_torchinfo=True,
    device=device
)

utils.save_model(
    model=student3,
    target_dir=STUDENT3_DIR,
    model_name="student3_learnable.pth"
)

print(f"✓ Student3 model saved to {STUDENT3_DIR}")
print(f"✓ Test Accuracy: {results3['test_top1_accuracy']:.2f}%")

all_results['Student3_Learnable'] = {
    'test_top1_accuracy': results3['test_top1_accuracy'],
    'test_top3_accuracy': results3['test_top3_accuracy'],
    'test_loss': results3['test_loss'],
    'training_time': student3_time,
    'ats_granularity': 'learnable',
    'ats_method': 'learnable_per_class',
    'save_dir': str(STUDENT3_DIR),
    'best_epoch': results3['best_epoch'],
    'best_val_loss': results3['best_val_loss']
}

# ============================================================================
# PHASE 5: TRAIN STUDENT4 WITH FUSION-ATS (STEP 4 - NEW)
# ============================================================================

print(f"\n{'='*100}")
print(f"PHASE 5: TRAINING STUDENT4 WITH FUSION-ATS-KD ")
print(f"{'='*100}\n")

print(f"Configuration:")
print(f" - Granularity: FUSION (Unified Multi-Granularity ATS)")
print(f" - Method: Combining Samplewise + Classwise + Learnable")
print(f" - T_BASE: {T_BASE}, T_MIN: {T_MIN}, T_MAX: {T_MAX}")
print(f" - Fusion Weights:")
print(f"   * Alpha (Samplewise): {FUSION_ALPHA}")
print(f"   * Beta (Classwise): {FUSION_BETA}")
print(f"   * Gamma (Learnable): {FUSION_GAMMA}\n")

student4 = model_builder.MobileNetV3Large(num_classes=len(class_names))
optimizer4 = torch.optim.Adam(student4.parameters(), lr=STUDENT_LR)

student4_start = timer()

results4 = KD_engine.train_and_test_with_KD(
    teacher=Teacher,
    student=student4,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    real_test_dataloader=real_test_dataloader,
    optimizer=optimizer4,
    loss_fn=loss_fn_student,
    epochs=STUDENT_EPOCHS,
    T_base=T_BASE,
    T_min=T_MIN,
    T_max=T_MAX,
    soft_target_loss_weight=1-ALPHA,
    ce_loss_weight=ALPHA,
    device=device,
    class_names=class_names,
    save_dir=STUDENT4_DIR,
    ats_method=ATS_METHOD,
    ats_granularity="fusion",
    fusion_weights=(FUSION_ALPHA, FUSION_BETA, FUSION_GAMMA),
    store_temperature_history=STORE_TEMPERATURE_HISTORY,
    num_classes=len(class_names)
)

student4_end = timer()
student4_time = student4_end - student4_start

print(f"\n✓ Student4 (Fusion ATS) training completed in {utils.format_time(student4_time)}")

utils.save_model_summary(
    student4,
    input_size=[32, 3, 224, 224],
    file_path=STUDENT4_DIR / "model_summary_student4.txt",
    use_torchinfo=True,
    device=device
)

utils.save_model(
    model=student4,
    target_dir=STUDENT4_DIR,
    model_name="student4_fusion.pth"
)

print(f"✓ Student4 model saved to {STUDENT4_DIR}")
print(f"✓ Test Accuracy: {results4['test_top1_accuracy']:.2f}%")

all_results['Student4_Fusion'] = {
    'test_top1_accuracy': results4['test_top1_accuracy'],
    'test_top3_accuracy': results4['test_top3_accuracy'],
    'test_loss': results4['test_loss'],
    'training_time': student4_time,
    'ats_granularity': 'fusion',
    'ats_method': f'fusion_({FUSION_ALPHA}, {FUSION_BETA}, {FUSION_GAMMA})',
    'save_dir': str(STUDENT4_DIR),
    'best_epoch': results4['best_epoch'],
    'best_val_loss': results4['best_val_loss']
}

# ============================================================================
# PHASE 6: GENERATE COMPARISON REPORT
# ============================================================================

print(f"\n{'='*100}")
print(f"PHASE 6: GENERATING COMPREHENSIVE COMPARISON REPORT")
print(f"{'='*100}\n")

# Create comparison DataFrame
comparison_data = {
    'Model': [
        'Teacher (Baseline)',
        'Student1 (Samplewise ATS)',
        'Student2 (Classwise ATS)',
        'Student3 (Learnable ATS)',
        'Student4 (Fusion ATS)'
    ],
    'Test Top-1 Accuracy (%)': [
        '-',
        f"{results1['test_top1_accuracy']:.2f}",
        f"{results2['test_top1_accuracy']:.2f}",
        f"{results3['test_top1_accuracy']:.2f}",
        f"{results4['test_top1_accuracy']:.2f}"
    ],
    'Test Top-3 Accuracy (%)': [
        '-',
        f"{results1['test_top3_accuracy']:.2f}",
        f"{results2['test_top3_accuracy']:.2f}",
        f"{results3['test_top3_accuracy']:.2f}",
        f"{results4['test_top3_accuracy']:.2f}"
    ],
    'Test Loss': [
        '-',
        f"{results1['test_loss']:.4f}",
        f"{results2['test_loss']:.4f}",
        f"{results3['test_loss']:.4f}",
        f"{results4['test_loss']:.4f}"
    ],
    'Training Time (s)': [
        f"{teacher_time:.2f}",
        f"{student1_time:.2f}",
        f"{student2_time:.2f}",
        f"{student3_time:.2f}",
        f"{student4_time:.2f}"
    ],
    'Best Epoch': [
        '-',
        f"{results1['best_epoch']}",
        f"{results2['best_epoch']}",
        f"{results3['best_epoch']}",
        f"{results4['best_epoch']}"
    ],
    'Best Val Loss': [
        '-',
        f"{results1['best_val_loss']:.4f}",
        f"{results2['best_val_loss']:.4f}",
        f"{results3['best_val_loss']:.4f}",
        f"{results4['best_val_loss']:.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("\nCOMPARISON TABLE:")
print(comparison_df.to_string(index=False))

# Save comparison as CSV
csv_path = COMPARISON_DIR / "model_comparison.csv"
comparison_df.to_csv(csv_path, index=False)

print(f"\n✓ Comparison table saved to {csv_path}")

# ================================================================
# ACCURACY COMPARISON
# ================================================================

print(f"\n{'='*50}")
print(f"ACCURACY COMPARISON")
print(f"{'='*50}")

top1_accs = [
    results1['test_top1_accuracy'],
    results2['test_top1_accuracy'],
    results3['test_top1_accuracy'],
    results4['test_top1_accuracy']
]

best_student_idx = np.argmax(top1_accs)
best_student_name = ['Samplewise', 'Classwise', 'Learnable', 'Fusion'][best_student_idx]
best_accuracy = top1_accs[best_student_idx]

print(f"\nBest Performing Student: {best_student_name} ATS")
print(f"Best Test Top-1 Accuracy: {best_accuracy:.2f}%")
print(f"Improvement over Samplewise: {best_accuracy - results1['test_top1_accuracy']:.2f}%")
print(f"Improvement over Classwise: {best_accuracy - results2['test_top1_accuracy']:.2f}%")
print(f"Improvement over Learnable: {best_accuracy - results3['test_top1_accuracy']:.2f}%")
print(f"Improvement over Fusion: {best_accuracy - results4['test_top1_accuracy']:.2f}%")

# ================================================================
# EFFICIENCY COMPARISON
# ================================================================

print(f"\n{'='*50}")
print(f"TRAINING EFFICIENCY COMPARISON")
print(f"{'='*50}")

print(f"\nStudent1 (Samplewise ATS): {student1_time:.2f}s")
print(f"Student2 (Classwise ATS): {student2_time:.2f}s")
print(f"Student3 (Learnable ATS): {student3_time:.2f}s")
print(f"Student4 (Fusion ATS): {student4_time:.2f}s")

fastest_idx = np.argmin([student1_time, student2_time, student3_time, student4_time])
fastest_name = ['Samplewise', 'Classwise', 'Learnable', 'Fusion'][fastest_idx]

print(f"\nFastest: {fastest_name} ATS")

# ================================================================
# SAVE DETAILED RESULTS
# ================================================================

print(f"\n{'='*50}")
print(f"SAVING DETAILED RESULTS")
print(f"{'='*50}\n")

# Save complete results as JSON
complete_results = {
    'experiment_name': 'Multi-Granularity ATS Knowledge Distillation with Fusion (STEP 4)',
    'timestamp': str(timer()),
    'configuration': {
        'teacher_epochs': TEACHER_EPOCHS,
        'student_epochs': STUDENT_EPOCHS,
        'batch_size': BATCH_SIZE,
        'teacher_lr': TEACHER_LR,
        'student_lr': STUDENT_LR,
        'T_base': T_BASE,
        'T_min': T_MIN,
        'T_max': T_MAX,
        'ats_method': ATS_METHOD,
        'alpha': ALPHA,
        'fusion_weights': {
            'alpha': FUSION_ALPHA,
            'beta': FUSION_BETA,
            'gamma': FUSION_GAMMA
        }
    },
    'dataset': {
        'num_classes': len(class_names),
        'train_samples': len(train_dataloader.dataset),
        'val_samples': len(test_dataloader.dataset),
        'test_samples': len(real_test_dataloader.dataset),
        'class_names': class_names
    },
    'results': {
        'teacher': {
            'training_time': teacher_time,
            'save_dir': str(TEACHER_DIR)
        },
        'student1_samplewise': results1,
        'student2_classwise': results2,
        'student3_learnable': results3,
        'student4_fusion': results4,
        'comparison': comparison_df.to_dict('list')
    }
}

json_path = COMPARISON_DIR / "complete_results.json"

with open(json_path, 'w') as f:
    json.dump(complete_results, f, indent=4)

print(f"✓ Complete results saved to {json_path}")

# Save text report
report_path = COMPARISON_DIR / "experiment_report.txt"

with open(report_path, 'w') as f:
    f.write(f"{'='*80}\n")
    f.write(f"MULTI-GRANULARITY ATS KNOWLEDGE DISTILLATION EXPERIMENT WITH FUSION (STEP 4)\n")
    f.write(f"{'='*80}\n\n")
    
    f.write(f"PHASE 1: TEACHER TRAINING\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Training Time: {utils.format_time(teacher_time)}\n")
    f.write(f"Save Directory: {TEACHER_DIR}\n\n")
    
    f.write(f"PHASE 2: STUDENT1 - SAMPLEWISE ATS\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Test Top-1 Accuracy: {results1['test_top1_accuracy']:.2f}%\n")
    f.write(f"Test Top-3 Accuracy: {results1['test_top3_accuracy']:.2f}%\n")
    f.write(f"Test Loss: {results1['test_loss']:.4f}\n")
    f.write(f"Training Time: {utils.format_time(student1_time)}\n")
    f.write(f"Best Epoch: {results1['best_epoch']}\n")
    f.write(f"Best Val Loss: {results1['best_val_loss']:.4f}\n")
    f.write(f"Save Directory: {STUDENT1_DIR}\n\n")
    
    f.write(f"PHASE 3: STUDENT2 - CLASSWISE ATS\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Test Top-1 Accuracy: {results2['test_top1_accuracy']:.2f}%\n")
    f.write(f"Test Top-3 Accuracy: {results2['test_top3_accuracy']:.2f}%\n")
    f.write(f"Test Loss: {results2['test_loss']:.4f}\n")
    f.write(f"Training Time: {utils.format_time(student2_time)}\n")
    f.write(f"Best Epoch: {results2['best_epoch']}\n")
    f.write(f"Best Val Loss: {results2['best_val_loss']:.4f}\n")
    f.write(f"Save Directory: {STUDENT2_DIR}\n\n")
    
    f.write(f"PHASE 4: STUDENT3 - LEARNABLE ATS\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Test Top-1 Accuracy: {results3['test_top1_accuracy']:.2f}%\n")
    f.write(f"Test Top-3 Accuracy: {results3['test_top3_accuracy']:.2f}%\n")
    f.write(f"Test Loss: {results3['test_loss']:.4f}\n")
    f.write(f"Training Time: {utils.format_time(student3_time)}\n")
    f.write(f"Best Epoch: {results3['best_epoch']}\n")
    f.write(f"Best Val Loss: {results3['best_val_loss']:.4f}\n")
    f.write(f"Save Directory: {STUDENT3_DIR}\n\n")
    
    f.write(f"PHASE 5: STUDENT4 - FUSION ATS (STEP 4 - NEW)\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Test Top-1 Accuracy: {results4['test_top1_accuracy']:.2f}%\n")
    f.write(f"Test Top-3 Accuracy: {results4['test_top3_accuracy']:.2f}%\n")
    f.write(f"Test Loss: {results4['test_loss']:.4f}\n")
    f.write(f"Training Time: {utils.format_time(student4_time)}\n")
    f.write(f"Best Epoch: {results4['best_epoch']}\n")
    f.write(f"Best Val Loss: {results4['best_val_loss']:.4f}\n")
    f.write(f"Save Directory: {STUDENT4_DIR}\n\n")
    f.write(f"Fusion Weights:\n")
    f.write(f"  - Alpha (Samplewise): {FUSION_ALPHA}\n")
    f.write(f"  - Beta (Classwise): {FUSION_BETA}\n")
    f.write(f"  - Gamma (Learnable): {FUSION_GAMMA}\n\n")
    
    f.write(f"COMPARISON SUMMARY\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Best Performer: {best_student_name} ATS with {best_accuracy:.2f}% accuracy\n")
    f.write(f"Fastest Training: {fastest_name} ATS\n")
    f.write(f"\nComparison Table:\n")
    f.write(comparison_df.to_string(index=False))

print(f"✓ Experiment report saved to {report_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

experiment_end = timer()
total_experiment_time = experiment_end - experiment_start

print(f"\n{'='*100}")
print(f"EXPERIMENT COMPLETE!")
print(f"{'='*100}\n")

print(f"Total Experiment Time: {utils.format_time(total_experiment_time)}")

print(f"\nResults Summary:")
print(f" - Teacher Model: {TEACHER_DIR}")
print(f" - Student1 (Samplewise): {STUDENT1_DIR} → {results1['test_top1_accuracy']:.2f}%")
print(f" - Student2 (Classwise): {STUDENT2_DIR} → {results2['test_top1_accuracy']:.2f}%")
print(f" - Student3 (Learnable): {STUDENT3_DIR} → {results3['test_top1_accuracy']:.2f}%")
print(f" - Student4 (Fusion): {STUDENT4_DIR} → {results4['test_top1_accuracy']:.2f}%")

print(f"\nComparison Report: {COMPARISON_DIR}")
print(f" - CSV: {csv_path}")
print(f" - JSON: {json_path}")
print(f" - Report: {report_path}\n")

print(f"{'='*100}\n")

# Baseline YOLOv11n Training Results

## Test Configuration

**Model**: YOLOv11n (2.59M parameters)
**Date**: 2025-12-17
**Device**: M4 Max (MPS backend)
**Dataset**: Fold 0 (video_1, video_2 for training | video_0 for validation)

### Training Parameters
- Epochs: 10 (test run)
- Batch size: 16
- Image size: 640x640
- Optimizer: Auto (AdamW)
- Learning rate: 0.01 (initial), 0.01 (final)
- Early stopping patience: 10
- AMP: Enabled

### Dataset Statistics
- Train images: 16,793
- Validation images: 6,708
- Classes: 1 (COTS - Crown-of-Thorns Starfish)

## Results

### Final Validation Metrics (Epoch 10)

| Metric | Value | Notes |
|--------|-------|-------|
| Precision | 0.620 | 62.0% - Good precision |
| Recall | 0.091 | 9.1% - Very low recall |
| mAP50 | 0.154 | 15.4% - Needs improvement |
| mAP50-95 | 0.078 | 7.8% - Baseline performance |

### Training Convergence

**Loss Progression (Epoch 1 → Epoch 10)**:

| Loss Type | Initial | Final | Improvement |
|-----------|---------|-------|-------------|
| Box Loss | 2.573 | 1.756 | -31.7% |
| Classification Loss | 16.08 | 1.802 | -88.8% |
| DFL Loss | 1.297 | 1.005 | -22.5% |

### Epoch-by-Epoch Performance

| Epoch | Precision | Recall | mAP50 | mAP50-95 | Training Time |
|-------|-----------|--------|-------|----------|---------------|
| 1 | 0.750 | 0.039 | 0.063 | 0.030 | 635s |
| 2 | 0.294 | 0.085 | 0.066 | 0.027 | 555s |
| 3 | 0.371 | 0.037 | 0.040 | 0.017 | 554s |
| 4 | 0.371 | 0.078 | 0.099 | 0.046 | 537s |
| 5 | 0.567 | 0.048 | 0.079 | 0.038 | 530s |
| 6 | 0.422 | 0.066 | 0.085 | 0.045 | 538s |
| 7 | 0.356 | 0.085 | 0.121 | 0.061 | 525s |
| 8 | 0.476 | 0.120 | 0.137 | 0.065 | 517s |
| 9 | 0.541 | 0.098 | 0.162 | 0.085 | 505s |
| 10 | 0.620 | 0.091 | 0.154 | 0.078 | 507s |

**Best Performance**: Epoch 9 (mAP50: 0.162, mAP50-95: 0.085)

## Observations

### Strengths
1. Training pipeline works correctly on M4 Max with MPS backend
2. Good convergence in classification loss (-88.8%)
3. Decent precision (62%) - model is conservative in predictions
4. Training speed: ~500-635s per epoch on M4 Max

### Issues
1. **Very low recall (9.1%)** - Missing most starfish
   - This is critical for F2 score which emphasizes recall
   - Likely due to:
     - Small object detection challenges
     - Default confidence threshold too high
     - Need for hyperparameter tuning
2. **Low mAP scores** - Substantial room for improvement
3. **Unstable recall** - Fluctuates significantly across epochs

### Expected Improvements for Phase 2

Based on winning solution techniques:
- Adjust hyperparameters: box=0.2, iou_t=0.3 (from defaults)
- Add augmentations: rotation, mixup, Transpose
- Increase epochs to 50-100 for proper convergence
- Implement SAHI for small object detection
- Lower confidence threshold for higher recall

## Training Artifacts

**Location**: `runs/train/yolo11n_fold02/`

Generated files:
- Model weights: `weights/best.pt`, `weights/last.pt`
- Training curves: `results.png`, `BoxF1_curve.png`, `BoxPR_curve.png`
- Validation visualizations: `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`
- Confusion matrix: `confusion_matrix.png`, `confusion_matrix_normalized.png`
- Full metrics: `results.csv`

## Next Steps (Phase 2)

1. Implement hyperparameter optimization (box=0.2, iou_t=0.3)
2. Add augmentations from winning solution
3. Train for 50 epochs across all 3 folds
4. Calculate F2 scores for cross-validation
5. Compare against Phase 1 baseline

## Conclusion

The baseline training pipeline is functional and produces valid results. The model shows promise with 62% precision but suffers from very low recall (9.1%), which is expected for an under-trained baseline. The low recall is the primary bottleneck for F2 score performance.

The training infrastructure is validated and ready for Phase 2 improvements.

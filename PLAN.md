# Great Barrier Reef COTS Detection - Project Plan & Progress

## Overview

Building a winning-level YOLOv11 pipeline adapted from the 1st place Kaggle solution.

**Timeline:** 1-2 weeks
**Hardware:** MacBook M4 Max (MPS backend)
**Target:** CV F2 > 0.70

---

## Phases

### ✅ Phase 0: Project Setup (Completed)
**Branch:** `feature/phase-0-setup`
**PR:** #1
**Status:** ✅ Merged

**Achievements:**
- Added ML dependencies (ultralytics, torch, sahi, albumentations, supervision)
- Created project structure (src/, configs/, reports/)
- Added CLAUDE.md with development guidelines
- Updated README.md

---

### ✅ Phase 1: Data Pipeline & Baseline (Completed)
**Branch:** `feature/phase-1-baseline`
**Target CV F2:** Any valid score
**Status:** ✅ Complete - Pipeline Validated

**Tasks:**
- [x] Data conversion to YOLO format
- [x] 3-fold video-based splits
- [x] Create dataset.yaml configs
- [x] Implement F2 score evaluation (winning team's algorithm)
- [x] Baseline YOLOv11n training script
- [x] Test training pipeline (10-epoch test run)
- [x] Document baseline performance

**Deliverables:**
- `src/data/prepare_yolo_format.py` ✅
- `src/data/create_folds.py` ✅
- `configs/dataset_fold_*.yaml` ✅ (fold 0, 1, 2)
- `src/evaluation/f2_score.py` ✅
- `src/training/train_baseline.py` ✅
- Baseline performance report ✅ `reports/baseline_results.md`

**Baseline Results (10-epoch test on Fold 0):**
- mAP50: 0.154 (15.4%)
- mAP50-95: 0.078 (7.8%)
- Precision: 0.620 (62.0%)
- Recall: 0.091 (9.1%) - **Critical bottleneck for F2**
- Training validated on M4 Max with MPS backend

---

### ⚠️ Phase 2: Optimized Detection (Needs Revision)
**Branch:** `feature/phase-2-optimized-detection`
**Target CV F2:** > 0.65
**Status:** ⚠️ Attempted - Performance Degraded (Needs Revision)

**What Was Attempted:**
- [x] Created optimized training script
- [x] Tested hyperparameter changes (box: 7.5→5.0)
- [x] Disabled HSV augmentation (per winning solution)
- [x] 15-epoch training run on Fold 0
- [x] Comprehensive analysis and documentation

**Results (15 epochs, Fold 0):**
- mAP50: 0.082 (-47% vs baseline) ⚠️
- mAP50-95: 0.040 (-49% vs baseline) ⚠️
- Precision: 0.499 (-19% vs baseline) ⚠️
- Recall: 0.052 (-43% vs baseline) ⚠️
- **Conclusion**: Worse than baseline

**Issues Encountered:**
1. MPS backend incompatibility with rotation+mosaic (TAL errors)
2. Removing HSV augmentation hurt generalization
3. Memory issues (OOM crash at epoch 15)
4. Training instability (40x slower after epoch 5)

**Deliverables:**
- `src/training/train_optimized.py` ✅
- `reports/phase2_analysis.md` ✅
- `reports/final_report/figures/` ✅ (5 visualizations)
- `src/visualization/generate_report_figures.py` ✅

**Phase 2 Revision - Ablation Study (Completed):**
- ✅ Created ablation study framework (8 experiments, 5 epochs each)
- ✅ Completed all 8 experiments (2025-12-22)
- ✅ Generated comprehensive analysis: `reports/ablation_study_analysis.md`
- ✅ Created visualizations: `reports/ablation_study/` (3 figures + CSV)
- ✅ Updated PLAN.md with ablation results

**Key Discoveries:**
1. **HSV augmentation is CRITICAL** (-34.3% when disabled) - explains Phase 2 failure!
2. **Box weight reductions all hurt** (-6.9% to -30.9%) - keep default 7.5
3. **Rotation hurts on MPS** (-12.7%) - keep disabled
4. **Mixup is neutral** (+0.1%) - safe to add

**Deliverables:**
- `src/training/ablation_study.py` ✅
- `src/training/train_single_ablation.py` ✅
- `src/evaluation/compare_ablation.py` ✅
- `reports/ablation_study_analysis.md` ✅
- `reports/ablation_study/ablation_*.png` ✅ (3 visualizations)
- `reports/ablation_study/ablation_results.csv` ✅

**Strategy Going Forward**: Keep baseline hyperparameters unchanged, extend training to 20-30 epochs

---

### ⏳ Phase 3: Temporal Post-Processing
**Target CV F2:** > 0.68
**Status:** Not Started

**Planned Tasks:**
- Supervision integration for detection filtering
- Attention area boosting (winning strategy)
- Optional: ByteTrack for object tracking

**Expected Deliverables:**
- `src/postprocessing/temporal_boost.py`
- `src/postprocessing/detection_filter.py`
- CV F2 > 0.68

---

### ⏳ Phase 4: Final Model & Submission
**Target CV F2:** > 0.70
**Status:** Not Started

**Planned Tasks:**
- Train on all 3 videos (no holdout)
- Efficient batch inference pipeline
- Submission format generation
- Kaggle submission

**Expected Deliverables:**
- `src/inference/predict.py`
- Final model weights
- Kaggle submission
- CV F2 > 0.70

---

### ⏳ Phase 5: Classification Re-scoring (Optional)
**Target CV F2:** > 0.72
**Status:** Not Started
**Condition:** Only if Phase 4 CV F2 < 0.70 and time permits

**Planned Tasks:**
- Crop extraction (OOF predictions)
- 7-bin IoU classification
- Score fusion

**Expected Impact:** +0.02-0.04 CV F2

---

## Scores Tracker

| Phase | Model | Image Size | Epochs | mAP50 | mAP50-95 | Precision | Recall | Status | Date |
|-------|-------|------------|--------|-------|----------|-----------|--------|--------|------|
| 1 (baseline) | YOLOv11n | 640 | 10 | **0.154** | **0.078** | **0.620** | **0.091** | ✅ Best | 2025-12-17 |
| 2 (optimized) | YOLOv11n | 640 | 15 | 0.082 | 0.040 | 0.499 | 0.052 | ⚠️ Worse | 2025-12-19 |

### Ablation Study Results (5 epochs, Fold 0) - 2025-12-22

| Experiment | mAP50 | mAP50-95 | Precision | Recall | vs Baseline | Key Finding |
|------------|-------|----------|-----------|--------|-------------|-------------|
| **mixup_0.1** | **0.126** | 0.051 | 0.435 | **0.128** | +0.1% ✅ | Neutral/slightly positive |
| **baseline_control** | **0.126** | **0.052** | 0.396 | 0.127 | 0.0% | Control |
| box_5.5 | 0.117 | 0.052 | 0.393 | 0.122 | -6.9% ⚠️ | Box reduction hurts |
| box_5.0 | 0.117 | 0.045 | 0.369 | 0.119 | -7.3% ⚠️ | Box reduction hurts |
| rotation_10 | 0.110 | 0.053 | 0.348 | 0.116 | -12.7% ⚠️ | Rotation hurts (MPS?) |
| box_6.0 | 0.107 | 0.048 | 0.341 | 0.110 | -14.9% ⚠️ | Box reduction hurts |
| box_6.5 | 0.087 | 0.035 | 0.314 | 0.090 | -30.9% ⚠️ | Large box reduction hurts |
| **no_hsv** | **0.083** | 0.032 | 0.339 | 0.085 | **-34.3% ❌** | **HSV CRITICAL** |

**Key Findings:**
- ✅ **Baseline remains best performer** (Phase 1 @ 10 epochs: 0.154 mAP50)
- ✅ **Ablation study validates hypothesis**: HSV removal caused -34.3% drop
- ⚠️ Phase 2 failed due to: HSV disabled (-34%) + box reduction (-31%) = -47% combined
- ⚠️ ALL box weight reductions hurt performance (7.5→6.5/6.0/5.5/5.0)
- ⚠️ Rotation augmentation hurts on MPS (-12.7%)
- ✅ Mixup augmentation is neutral (+0.1%)
- 📊 All trained on Fold 0 only (video_1+2 train, video_0 val)
- 🎯 **Current best: Phase 1 baseline** (use for Phase 3+)
- 📈 **Recommendation**: Keep baseline hyperparameters, extend to 20-30 epochs

---

## Key Techniques (from 1st Place Solution)

| Technique | Status | Phase | Impact | Notes |
|-----------|--------|-------|--------|-------|
| 3-fold CV by video_id | ✅ Implemented | 1 | Foundation | Working |
| YOLO format conversion | ✅ Implemented | 1 | Required | 23,501 images |
| HSV augmentation | ✅ Critical | 1 | **Important** | Removing it hurt performance! |
| Modified hyperparams (box=0.2) | ⚠️ Failed | 2 | TBD | MPS compatibility issues |
| Rotation/Mixup augmentation | ⚠️ Blocked | 2 | TBD | Causes TAL errors on MPS |
| SAHI for small objects | ⏳ Planned | 2-rev | +5-8% | Next priority |
| Temporal boosting | ⏳ Planned | 3 | +2-4% | - |
| Classification re-scoring | ⏳ Optional | 5 | +2-4% | If needed |

---

## Git Workflow

- **Branch naming:** `feature/phase-N-description`
- **PR naming:** `Phase N: Description`
- **Merge strategy:** Squash and merge
- **Branch cleanup:** Auto-delete after merge
- **No "contributed by claude"** in commits

---

## References

- [Competition](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)
- [Winning Solution](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/qns-trust-cv-1st-place-solution)
- [F2 Score Algorithm](https://www.kaggle.com/haqishen/f2-evaluation/script)

---

**Last Updated:** 2025-12-17
**Current Phase:** Phase 1 - Data Pipeline & Baseline

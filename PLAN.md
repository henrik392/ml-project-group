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

### ⏳ Phase 2: Optimized Detection
**Target CV F2:** > 0.65
**Status:** Not Started

**Planned Tasks:**
- Model scaling (YOLOv11s, YOLOv11m)
- Higher resolution training (1280×720)
- Hyperparameter optimization (box=0.2, iou_t=0.3)
- Underwater augmentations (MotionBlur, RGBShift, CLAHE)
- SAHI integration for small objects

**Expected Deliverables:**
- `src/training/train_optimized.py`
- `configs/yolo_optimized.yaml`
- CV F2 > 0.65

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

| Phase | Model | Image Size | Epochs | mAP50 | mAP50-95 | Precision | Recall | Date |
|-------|-------|------------|--------|-------|----------|-----------|--------|------|
| 1 (baseline test) | YOLOv11n | 640 | 10 | 0.154 | 0.078 | 0.620 | 0.091 | 2025-12-17 |

**Note:** Phase 1 was a 10-epoch test run on Fold 0 only to validate the training pipeline. Full 3-fold CV training pending Phase 2.

---

## Key Techniques (from 1st Place Solution)

| Technique | Status | Phase | Impact |
|-----------|--------|-------|--------|
| 3-fold CV by video_id | ✅ Implemented | 1 | Foundation |
| YOLO format conversion | ✅ Implemented | 1 | Required |
| Modified hyperparams (box=0.2, iou_t=0.3) | ⏳ Planned | 2 | +5-10% |
| Underwater augmentations | ⏳ Planned | 2 | +3-5% |
| SAHI for small objects | ⏳ Planned | 2 | +5-8% |
| Temporal boosting | ⏳ Planned | 3 | +2-4% |
| Classification re-scoring | ⏳ Optional | 5 | +2-4% |

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

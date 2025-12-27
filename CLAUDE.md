# Great Barrier Reef – COTS Detection (V2)

## Project Context

* **Task**: Detect Crown-of-Thorns Starfish (COTS) in underwater **video**
* **Dataset**: TensorFlow Great Barrier Reef (Kaggle, 2022)
* **Metric**: **F2 score** (recall-weighted)
* **Goal**: High-quality **academic report** (single source of truth → slides + poster)

## Evaluation Protocol (CRITICAL)

* **Video-aware evaluation only** (no frame leakage)
* **Leave-one-video-out** (3 videos total):

  * Train on 2 videos, evaluate on the 3rd
* All results reported in **F2**
* Confidence thresholds optimized **for F2**

## Tech Stack & Rules

* **YOLOv11** via Ultralytics
* **SAHI** for tiled inference (optional experiment)
* **ByteTrack** via `model.track`
  → used as **temporal recall enhancement**, not ID tracking
* **No Ray Tune**
* **No Kaggle submissions**
* **No custom OpenCV preprocessing**

  * OpenCV allowed **only for visualization / video grids**
* Run Python via **`uv run` only**
* Train on **M4 Max (MPS)** or GPU if needed

---

## Scientific Writing Standard (MANDATORY)

### Always separate **Facts** from **Hypotheses**

**Facts (measured):**

* Empirical results from experiments
* Quantitative metrics (F2, recall, mAP, FPS)
* Observed training / validation behavior

**Language**:

> “We observed…”, “Results show…”, “The ablation demonstrates…”

**Hypotheses (unverified):**

* Explanations without direct evidence
* Assumptions about causality

**Language**:

> “We hypothesize…”, “This may suggest…”, “One possible explanation…”

### Required structure for analysis sections

1. **Observation** (fact)
2. **Hypothesis** (clearly labeled)
3. **Evidence** (if available)
4. **Next experiment / limitation**

---

## Experimental Story (Report Structure)

### 1) Historical Baseline

* **YOLOv5 (2022 SOTA)** vs **YOLOv11 (modern)**
* Same split, same metric, same protocol

### 2) YOLOv11 Ablation (Minimal)

* Only high-impact factors:

  * confidence threshold
  * NMS IoU
  * image resolution
* Goal: establish a **strong, defensible baseline**

### 3) Inference Strategy

* Standard inference vs **SAHI**
* Report **accuracy vs speed trade-off**

### 4) Temporal Context

* Add **ByteTrack**
* Motivation: video continuity improves **recall** under F2
* Also test **SAHI + ByteTrack**

### 5) Qualitative Analysis

* Side-by-side visual comparisons
* Failure cases (small / occluded starfish)

---

## Deliverables (Produced Continuously)

### Report (Primary Artifact)

* **Methods**: model, data split, evaluation
* **Results**: tables (F2, recall, FPS)
* **Analysis**: ablations, trade-offs, limitations
* **Figures**: clean, publication-ready

### Visuals

* PNG grids for report
* One **4-up MP4** comparison video:

  * YOLOv5 | YOLOv11 | SAHI | SAHI + ByteTrack

---

## File Expectations

```
reports/
├── final_report.md
├── figures/
│   ├── performance_tables.png
│   ├── ablation_results.png
│   └── example_predictions.png
videos/
└── comparison_4up.mp4
metrics/
└── results.csv
```

---

## Key Principle

> **Clarity > complexity.**
> If YOLOv11 defaults outperform tweaks, we **report that and move on**.

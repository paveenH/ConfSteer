# ConfSteer Result Log

---

## 2026-03-20 — Neutral Role Only

> Training data: **neutral role only** — single H5 per task, no role variation.

## 1. Data Collection & Label Generation

### Prompt Format

All answers are collected using the following prompt template:

```
Would you answer the following question with A, B, C or D?
Question: {question}
A) ...  B) ...  C) ...  D) ...
Your answer among "A, B, C, D" is:
```

The hidden state at the **last token** of this prompt is extracted across all layers and saved to `.h5`.
Steered answers (`mdf_4`, `mdf_-4`) use the **same prompt** with RSN diff vectors injected into the residual stream at specified layers during the forward pass — the prompt itself does not change.

### Label Generation (`make_labels.py`)

For each sample, three forward passes are compared:

| Answer file | Description |
|---|---|
| `orig/` | Original unsteered answer |
| `mdf_4/` | Answer after +4 alpha steering |
| `mdf_-4/` | Answer after -4 alpha steering |

### Dataset Overview (llama3-8B, 22,787 total samples)

| Class | Count | Percentage |
|---|---|---|
| `label_pos4 == +1` (steering helps) | 1,305 | 5.7% |
| `label_neg4 == +1` (neg steering helps) | 626 | 2.7% |
| All others (no improvement) | 20,856 | 91.6% |

Tasks: MMLU-Pro (60+ subjects), FACTOR, AR-LSAT, GPQA, LogiQA, TruthfulQA MC1/MC2
Hidden state shape: `(N, 33 layers, 4096 dims)` — last token, all layers

---

## 2. 3-Class Classification (`classifier_demo.py`)

### Label Definition

| Classifier class | Condition | Meaning |
|---|---|---|
| `+1` | `label_pos4 == 1` | Apply +4 steering |
| `-1` | `label_neg4 == 1` | Apply -4 steering |
| `0` | all others | No steering |

Conflict case (`label_pos4 == label_neg4 == 1`) assigned to `+1`.

### Sampling

- Classes `+1` and `-1` kept in full
- Class `0` downsampled to `1.0 × max(|+1|, |-1|) = 1305`
- Final: `+1`: 1305, `-1`: 626, `0`: 1305 → **Total: 3,236**

### Pipeline

```
Hidden states (N, 4096)
  → StandardScaler
  → [optional] PCA(n_components)
  → LogisticRegression(class_weight="balanced", C=1.0)
  → train_test_split(test=0.2)
  → classification_report
```

### Results (llama3-8B, Layer 25, PCA=200)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| neg(-1) | 0.33 | 0.48 | 0.39 |
| no_change(0) | 0.59 | 0.46 | 0.52 |
| pos(+1) | 0.59 | 0.59 | 0.59 |
| **macro avg** | **0.50** | **0.51** | **0.50** |

Accuracy: 0.52 (random baseline: 0.33)

**Limitations:**
- `neg(-1)` class only 626 samples → weakest performance (F1=0.39)
- High-dimensional features (4096) relative to sample size cause noise
- PCA(200, 85.8% variance) provides marginal improvement over raw features

---

## 3. Binary Classification (`classifier_binary.py`)

### Motivation

The `neg(-1)` class (626 samples) is too small to learn reliably. Dropping it and reframing as a binary problem: **should we apply +4 steering?**

### Label Definition

| Classifier class | Condition | Meaning |
|---|---|---|
| `1` (steer) | `label_pos4 == 1` | +4 steering improves answer |
| `0` (no_steer) | all others | includes neg(-1), no change, already correct |

`label_neg4` is not used — all non-`label_pos4==1` samples are treated as "do not steer".

### Sampling

- Class `1` kept in full: **1,305**
- Class `0` downsampled to `1.0 × 1305 = 1305`
- Final: **2,610 total**

### Pipeline

```
Hidden states (N, 4096)  [single layer]
  → StandardScaler
  → PCA(50)               [~40% variance, noise removal]
  → LogisticRegression(class_weight="balanced", C=1.0)
  → StratifiedKFold(5)   → CV F1, CV AUC
  → train_test_split(test=0.2) → hold-out evaluation
```

### Layer Sweep Results (llama3-8B, PCA=50, all 33 layers)

Key findings from sweeping all layers:

| Layer range | CV F1 | CV AUC | Notes |
|---|---|---|---|
| Layer 0 | 0.000 | 0.500 | Embedding layer — no signal |
| Layers 1–14 | 0.625–0.647 | 0.636–0.666 | Gradually increasing |
| Layers 15–25 | 0.656–0.677 | 0.675–0.696 | Best region |
| Layers 26–32 | 0.653–0.669 | 0.688–0.696 | Slight decline |

**Best layer: 25** — F1=0.677 ± 0.018, AUC=0.696

Full sweep:

| Layer | CV F1 | CV AUC |
|---|---|---|
| 0 | 0.000 | 0.500 |
| 1 | 0.627 | 0.636 |
| 7 | 0.642 | 0.653 |
| 15 | 0.656 | 0.675 |
| 18 | 0.651 | 0.684 |
| 19 | 0.654 | 0.683 |
| 22 | 0.666 | 0.688 |
| 23 | 0.670 | 0.693 |
| **25** | **0.677** | **0.696** |
| 28 | 0.667 | 0.696 |
| 32 | 0.653 | 0.688 |

### Learning Curve (Layer 25, PCA=50)

Val F1 from n=1040 → n=2600: **0.62 → 0.67** (nearly flat)

**Interpretation: signal bottleneck, not data bottleneck.**
Adding more samples does not improve performance — the ceiling is determined by the discriminative information available in the hidden state at this layer.

### Summary

| Metric | Value |
|---|---|
| Best layer | 25 (≈ 75% depth of 33-layer model) |
| CV F1 (steer class) | 0.677 ± 0.018 |
| CV AUC | 0.696 |
| Signal distribution | Layers 15–25 carry most information |
| Data bottleneck? | No — learning curve is flat |
| Signal ceiling | AUC ≈ 0.70 for single-layer LR |

**Next direction:** Aggregate information across multiple layers (mean pooling / weighted sum / attention) to potentially exceed the single-layer AUC ceiling.

---

## 2026-03-26 — All Roles

> Training data: **all 7 roles** (`neutral`, `confident`, `unconfident`, `expert`, `non_expert`, `student`, `person`)
> Samples from: `samples/{model}/samples_binary_all.npz` (generated by `prepare_samples.py`, ratio=1.0)

### Dataset Overview

| Model | Total layers | Class 1 (steer) | Class 0 (no_steer) | Total |
|---|---|---|---|---|
| llama3-8B | 33 | 11,607 | 11,607 | 23,214 |
| qwen3-8B | 37 | 10,232 | 10,232 | 20,464 |

---

> **Known caveat**: train/test split is sample-level, not question-level — the same question may appear in both splits across roles, potentially inflating AUC. Applies equally to all classifiers; relative comparisons remain valid. Will be corrected with question-level split.

### Results Summary

| Model | Classifier | Layers used | CV F1 | CV AUC | Test Acc | Test AUC | F1 macro |
|---|---|---|---|---|---|---|---|
| llama3-8B | LR | 19 | 0.698 ± 0.006 | 0.750 | 0.68 | 0.748 | 0.68 |
| llama3-8B | LR | 25 | 0.706 ± 0.007 | 0.756 | 0.70 | 0.750 | 0.70 |
| llama3-8B | LR | 32 (last) | 0.700 ± 0.003 | 0.754 | 0.69 | 0.750 | 0.69 |
| llama3-8B | CNN | all (0–32) | — | — | 0.75 | 0.821 | 0.75 |
| qwen3-8B | LR | 25 | 0.706 ± 0.004 | 0.768 | 0.70 | 0.765 | 0.70 |
| qwen3-8B | LR | 36 (last) | 0.715 ± 0.005 | 0.775 | 0.71 | 0.784 | 0.71 |
| qwen3-8B | CNN | all (0–36) | — | — | 0.72 | 0.804 | 0.72 |

**Observations**:
- CNN outperforms single-layer LR for both models
- llama3 benefits more from CNN (+0.07 AUC vs layer 25 LR) — steering signal is distributed across layers
- qwen3 CNN gain is smaller (+0.02 vs last layer LR) — signal already concentrated in the last few layers
- qwen3 LR last layer (0.784) ≈ llama3 CNN all layers (0.821), suggesting qwen3 representations are more linearly separable

### Layer Sweep (PCA=50, for layer selection reference)

| Model | Best layer (F1) | Best layer (AUC) | Plateau range | Note |
|---|---|---|---|---|
| llama3-8B | 19 | 23 | 18–25 | No-PCA favors layer 25 |
| qwen3-8B | 25 | 32 | 25–36 | Sharp rise at layer 21→22 |

AUC with PCA=50 is ~0.06 lower than no-PCA — sweep used for relative comparison only.

### CNN Training Details

Architecture: Linear proj (D→64) → 1D-CNN (kernel=3, ch=64) → Layer Attention → MLP → 2 classes
Training: 20 epochs, AdamW lr=1e-3, CosineAnnealingLR, batch=64, dropout=0.5, best checkpoint by val loss

| Model | Best epoch | Val Loss | Val Acc |
|---|---|---|---|
| llama3-8B | 7 | 0.534 | 0.750 |
| qwen3-8B | 5 | 0.531 | 0.721 |

---

## 2025-03-26 — Resample (Question-Level Split)

> Re-split by question ID `(task, orig_stem, index)` to eliminate data leakage across roles.
> Train set: downsampled 1:1. Test set: original class distribution (~7% steer, ~93% no_steer).

### Dataset (question-level split)

| Model | Train (1:1) | Test steer(1) | Test no_steer(0) | Test total |
|---|---|---|---|---|
| llama3-8B | 18,624 | 2,295 | 29,611 | 31,906 |
| qwen3-8B | 16,378 | 2,043 | 29,863 | 31,906 |

### Results Summary

| Model | Classifier | Layers | CV AUC | Test AUC | Precision (steer) | Recall (steer) | FP | TP |
|---|---|---|---|---|---|---|---|---|
| llama3-8B | LR | 25 | 0.770 | 0.654 | 0.11 | 0.52 | 9,318 | 1,194 |
| qwen3-8B | LR | 36 | 0.771 | 0.670 | 0.10 | 0.52 | 9,146 | 1,067 |
| llama3-8B | CNN | all (0–32) | — | 0.692 | 0.14 | 0.32 | 4,551 | 745 |
| qwen3-8B | CNN | all (0–36) | — | 0.549 | 0.08 | 0.08 | 1,872 | 156 |

> **Note:** CV AUC is computed on the balanced train set (1:1); Test AUC is on the imbalanced test set.
> Compared to sample-level split, Test AUC drops ~0.10 for LR — confirming prior results were inflated by leakage.

### Threshold Sweep (llama3 LR layer 25, CV AUC=0.770, Test AUC=0.654)

| Threshold | Precision (steer) | Recall (steer) | TP | FP | TP/FP |
|---|---|---|---|---|---|
| 0.5 | 0.11 | 0.52 | 1,194 | 9,318 | 1:7.8 |
| 0.7 | 0.12 | 0.41 | 950 | 7,195 | 1:7.6 |
| 0.8 | 0.12 | 0.36 | 820 | 5,967 | 1:7.3 |

Raising the threshold reduces both TP and FP proportionally — precision stays flat at 0.11–0.12. The score distribution has no high-precision region; threshold tuning cannot rescue a low-AUC model.

### qwen3 CNN anomaly

qwen3 CNN AUC=0.549 (near random), with epoch 1 val acc=6.5% (near-zero). Training loss barely moves (0.694→0.673). qwen3 architecture confirmed: 37 layers (including embedding), hidden dim=4096 — README has been corrected.

### Observations

- **Leakage confirmed**: LR Test AUC 0.65–0.67 vs sample-level ~0.75–0.78 → ~0.10 inflation from question leakage
- **CNN vs LR (llama3)**: CNN AUC 0.692 vs LR 0.654 (+0.038); CNN is more conservative (higher precision, lower recall)
- **Precision bottleneck**: steer precision 0.10–0.14 regardless of threshold → net accuracy gain unlikely in real deployment
- **Signal ceiling**: AUC ~0.69 for llama3; predicting steering effectiveness from hidden states appears fundamentally limited
- **Next direction**: switch target to `orig_correct` (is the model currently right?) — stronger signal, better supported by probing literature

---

## 2026-03-27 — orig_correct Classification

> Target: **is the model's original (unsteered) answer correct?**
> Label: `y=1` (correct), `y=0` (wrong) — derived from `orig_correct` field in label files.
> Training data: all 7 roles, question-level split.
> Train: 1000 per class (max_per_class=1000). Test: 1000 per class (max_test_per_class=1000).

### Dataset Overview

| Model | Train correct(1) | Train wrong(0) | Test correct(1) | Test wrong(0) |
|---|---|---|---|---|
| llama3-8B | 1,000 | 1,000 | 1,000 | 1,000 |
| qwen3-8B | — | — | — | — |

> Raw counts before sampling — llama3: train 51,669 correct / 75,934 wrong; test 12,779 correct / 19,127 wrong.

### Results

| Model | Classifier | Layer | Samples (train/test) | CV AUC | Test AUC | Test Acc | F1 macro |
|---|---|---|---|---|---|---|---|
| llama3-8B | LR | 22 (best F1) / 23 (best AUC) | 1k/1k | 0.677 / 0.732 | — | — | — |
| llama3-8B | CNN | all (0–32) | 1k/1k | — | 0.736 | 0.69 | 0.69 |
| llama3-8B | CNN | all (0–32) | 5k/2k | — | 0.768 | 0.70 | 0.70 |
| qwen3-8B | LR | — | — | — | — | — | — |
| qwen3-8B | CNN | all | — | — | — | — | — |

> **llama3 LR layer sweep (1k)**: Signal rises from layer 1 (AUC=0.657) and peaks at layer 21–23 (AUC=0.727–0.732). Layers 21–32 form a plateau (~0.72–0.73). Best layer by AUC: **23** (0.732), by F1: **22** (0.677). Notably higher than steering prediction at same layer (AUC ~0.70) — confirms orig_correct is a stronger signal.

> **llama3 CNN (1k)**: Best checkpoint epoch 1 (val_loss=0.622). Val loss diverges from epoch 2 onward — classic overfitting on small train set. Train acc reaches 0.821 by epoch 20 while val acc stays ~0.69, confirming data bottleneck.

---

### orig_correct vs. Steering Effectiveness Analysis (`analyze_orig_vs_steering.py`)

> All 7 roles, all tasks, no sampling — full label set (159,509 samples each model).

#### Raw Statistics

| | llama3-8B | qwen3-8B |
|---|---|---|
| Total samples | 159,509 | 159,509 |
| orig_correct=1 (right) | 64,448 (40.4%) | 77,527 (48.6%) |
| orig_correct=0 (wrong) | 95,061 (59.6%) | 81,982 (51.4%) |
| +4 fixes wrong (wrong→correct) | 11,607 / 95,061 = **12.2%** | 10,232 / 81,982 = **12.5%** |
| +4 harms correct (correct→wrong) | 8,959 / 64,448 = **13.9%** | 7,720 / 77,527 = **10.0%** |
| Steering effective rate (overall) | 7.3% | 6.4% |
| Steering effective rate (wrong only) | 12.2% | 12.5% |
| Lift from targeting wrong samples | **1.68×** | **1.95×** |

#### Simulation

| Scenario | llama3-8B | qwen3-8B |
|---|---|---|
| Steer ALL wrong samples → net acc change | +7.28% | +6.41% |
| Steer ALL samples → net acc change | +1.66% | +1.57% |

#### Analysis

- **Targeting wrong samples helps**: steering effective rate doubles from ~7% (random) to ~12% when restricted to wrong samples — 1.68× lift for llama3, 1.95× for qwen3.
- **Upper bound if classifier is perfect**: if orig_correct classifier achieves 100% recall on wrong samples with 0 false positives, net accuracy gain is **+7.3% (llama3) / +6.4% (qwen3)**.
- **The cost of false positives**: steering a correct sample has a 13.9% (llama3) / 10.0% (qwen3) chance of breaking it. With our classifier at AUC~0.73, false positives are inevitable — each one carries this harm risk.
- **Practical ceiling**: with classifier recall~0.70 and precision~0.65 (estimated from AUC=0.73), expected net gain is roughly `0.70×7.3% − FP_rate×13.9%`. Whether this is positive depends on the operating threshold.
- **qwen3 is a better candidate**: lower harm rate on correct samples (10.0% vs 13.9%) and higher lift (1.95×) make qwen3 more suitable for this steering strategy.

---

### Classifier Benchmark (llama3-8B)

> Question-level split; per-layer StandardScaler; all layers (0–32); balanced classes (y=0/y=1 equal).
> Train/test sizes: **5k** = 10k/4k samples; **25k** = 50k/4k samples.

| Classifier | Train Size | Architecture | Best Epoch | Test AUC | Test Acc | F1 macro | Notes |
|---|---|---|---|---|---|---|---|
| LR (layer 23) | 1k | Single-layer LR | — | 0.732 | — | — | CV AUC; single best layer |
| CNN | 1k | Linear proj (D→64) → 1D-CNN → LayerAttn → MLP | 1 | 0.736 | 0.69 | 0.69 | Overfits from epoch 2 |
| CNN | 5k | Linear proj (D→64) → 1D-CNN → LayerAttn → MLP | 1 | 0.768 | 0.70 | 0.70 | Overfits from epoch 2 |
| PCA-CNN | 5k | Per-layer PCA (D→128) → 1D-CNN → LayerAttn | 3 | 0.770 | 0.71 | 0.71 | PCA variance = nan (numerical, non-critical) |
| L1-MLP | 5k | L1 sparse selector (top-1024) → MLP(256→64→2) | 1 | 0.775 | 0.71 | 0.71 | Stage 2 still overfits |
| Sparse Attn | 5k | Dim-proj (L→64) → Top-k (k=512) → MLP | 29 | 0.746 | 0.69 | 0.69 | No overfitting; train/val converge |
| Sparse Attn | 5k | Dim-proj (L→64) → Top-k (k=1024) → MLP | 27 | 0.750 | 0.68 | 0.68 | topk only affects inference, not param count |
| 2D-CNN | 5k | Conv2d(1→32→64, kernel=3×64) → GAP → MLP | 30 | 0.729 | 0.67 | 0.67 | Slow convergence; 403k params |
| PCA-CNN | **25k** | Per-layer PCA (D→128) → 1D-CNN → LayerAttn | 2 | **0.786** | 0.72 | 0.72 | +0.016 vs 5k; overfitting persists |
| Sparse Attn | **25k** | Dim-proj (L→64) → Top-k (k=512) → MLP | 27 | 0.757 | 0.69 | 0.69 | +0.011 vs 5k; no overfitting |
| L1-CNN | **25k** | Per-layer L1 (D→256) → 1D-CNN → LayerAttn | 3 | 0.774 | 0.70 | 0.70 | topd=256; overfits from epoch 4 |
| Transformer | **25k** | Per-layer PCA (D→128) → TransformerEnc (2L, nhead=4) → MeanPool | 2 | 0.780 | 0.71 | 0.71 | 294k params; overfits from epoch 3 |

**Key Observations:**

1. **Overfitting is universal** — all models best at epoch 1–3; train acc → 100%, val stagnates. Architecture and regularization alone do not resolve this.
2. **Bias-variance trade-off** — Sparse Attn (10,947 params) is the only model that doesn't overfit, but underfits instead (AUC 0.757 vs 0.780–0.786). The signal requires sufficient capacity to capture.
3. **Scaling helps** — 5k→25k raises AUC by +0.011–0.016 across all architectures.
4. **Transformer vs PCA-CNN** — Transformer (294k params) captures non-local cross-layer interactions but AUC 0.780 < PCA-CNN 0.786; more capacity does not help here.
5. **Current best: PCA-CNN @ 25k → AUC 0.786**

**Pending:** qwen3 experiments; middle-layers-only (11–19) ablation; RSN neuron projection features.


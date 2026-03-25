# ConfSteer Classification Results Summary

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

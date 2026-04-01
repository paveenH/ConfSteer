# ConfSteer — Standard Operating Procedure (llama3-8B)

This document describes the end-to-end pipeline for training and evaluating a classifier-guided steering system.
The pipeline spans two repositories:

| Repo | Path | Purpose |
|---|---|---|
| **RolePlaying** | `/data1/paveen/RolePlaying` | Model inference, hidden state extraction, steering |
| **ConfSteer** | `/data1/paveen/ConfSteer` | Classifier training and OOD evaluation |

---

## Overview

```
Step 1  Extract hidden states + original answers     [RolePlaying]
Step 2  Generate steered answers (mdf)               [RolePlaying]
Step 3  Prepare training samples (.npz)              [ConfSteer]
Step 4  Train PCA-CNN classifier                     [ConfSteer]
Step 5  Evaluate OOD generalization                  [ConfSteer]
Step 6  Run classifier-guided steering benchmark     [RolePlaying]
```

---

## Step 1 — Extract Hidden States + Original Answers

**Repo**: RolePlaying  
**Purpose**: Run llama3 on each benchmark task with 7 role prompts, save per-layer hidden states (H5) and answer logits (JSON).

### Roles (7)
```
neutral, {task} expert, non {task} expert, confident, unconfident, student, person
```
> For MMLU-Pro, `student` becomes `{task} student` (task-prefixed).

### MMLU

```bash
# In: /data1/paveen/RolePlaying
bash run_hidden_mmlu.sh llama3
```

**Script**: `run_hidden_mmlu.sh` → calls `get_answer_logits.py --save`

**Output**:
```
ConfSteer/HiddenStates/llama3/mmlu/{role}_{task}_8B.h5
RolePlaying/components/llama3/answer_hs_mmlu/{task}_8B_answers.json
```

### MMLU-Pro (and other benchmarks)

For MMLU-Pro, GPQA, FACTOR, AR-LSAT, LogiQA — use `get_answer_logits_mmlupro.py` directly or via the benchmark shell scripts.

```bash
# Example: MMLU-Pro, neutral role
python get_answer_logits_mmlupro.py \
    --model llama3 \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --size 8B \
    --type non \
    --test_file benchmark/mmlupro_test.json \
    --ans_file answer_mmlupro \
    --roles "neutral,{task} expert,non {task} expert,confident,unconfident,{task} student,person" \
    --suite default \
    --base_dir components \
    --data data1 \
    --save
```

**Output**:
```
ConfSteer/HiddenStates/llama3/mmlupro/{role_prefix}_{task}_8B.h5
RolePlaying/components/llama3/answer_mmlupro/orig/{task}_8B_answers.json
```

> H5 naming convention: prefix uses **lowercased** task slug (e.g. `electroniccommunications_expert_ElectronicCommunications_8B.h5`).

---

## Step 2 — Generate Steered Answers (mdf)

**Repo**: RolePlaying  
**Purpose**: Re-run inference with RSN diff vectors injected into the residual stream. Produces `mdf_4` (α=+4) and `mdf_-4` (α=−4) answer files used to label which samples benefit from steering.

### llama3-8B recommended config
| Parameter | Value |
|---|---|
| alpha | 4 / −4 |
| layers | 11–20 |
| mask_type | nmd |
| percentage | 0.5 |

### MMLU
```bash
# In: /data1/paveen/RolePlaying
bash run_benchmark_llama3.sh   # runs get_answer_regenerate_logits.py for all roles
```

### MMLU-Pro and other benchmarks
```bash
bash run_benchmark_llama3.sh   # also covers mmlupro, factor, gpqa, arlsat, logiqa
```

**Script**: calls `get_answer_regenerate_logits_mmlupro.py`

**Output**:
```
RolePlaying/components/llama3/answer_mdf_mmlu/mdf_4/{task}_8B_answers.json
RolePlaying/components/llama3/answer_mdf_mmlu/mdf_-4/{task}_8B_answers.json
RolePlaying/components/llama3/answer_mdf_mmlupro/mdf_4/{task}_8B_answers.json
RolePlaying/components/llama3/answer_mdf_mmlupro/mdf_-4/{task}_8B_answers.json
```

> Step 2 outputs are used for downstream steering benchmarks (Step 6). They are **not required** for classifier training (Steps 3–5), which only uses `orig` answers and hidden states.

---

## Step 3 — Prepare Training Samples

**Repo**: ConfSteer  
**Purpose**: Read original answer JSONs + H5 hidden states, assign `orig_correct` labels (1=correct, 0=wrong), apply question-level train/test split, and save `.npz` files.

**Label**: `y=1` if the model's original (unsteered) answer matches the ground truth; `y=0` otherwise.  
**Split**: All 7 roles of the same question always go to the same split (no leakage across roles).

### MMLU
```bash
# In: /data1/paveen/ConfSteer
python prepare_samples_mmlu.py --model llama3 --size 8B
```

**Output**:
```
samples/llama3/samples_orig_mmlu_all_train.npz
samples/llama3/samples_orig_mmlu_all_test.npz
```

### MMLU-Pro
```bash
python prepare_samples_mmlupro.py --model llama3 --size 8B
```

**Output**:
```
samples/llama3/samples_orig_mmlupro_all_train.npz
samples/llama3/samples_orig_mmlupro_all_test.npz
```

### NPZ format
Each `.npz` contains:
| Key | Shape | dtype | Description |
|---|---|---|---|
| `X` | (N, 33, 4096) | float16 | Hidden states — all layers, last token |
| `y` | (N,) | int8 | 1=correct, 0=wrong |
| `meta` | (N,) | str (JSON) | `{"task", "role", "index"}` per sample |
| `roles` | (R,) | str | List of roles included |

---

## Step 4 — Train PCA-CNN Classifier

**Repo**: ConfSteer  
**Purpose**: Train a binary classifier on hidden states to predict `orig_correct`. Uses per-layer PCA (4096→128) followed by a 1D-CNN with residual connection and layer attention.

### Architecture
```
Input (N, 33, 4096)
  → Per-layer StandardScaler + PCA (4096 → 128)
  → Conv1d (k=7) + residual → Layer Attention
  → MLP → 2 classes (correct / wrong)
```

### Training (single benchmark)
```bash
# In: /data1/paveen/ConfSteer
python classifier_pca_cnn.py \
    --model llama3 \
    --train samples/llama3/samples_orig_mmlu_all_train.npz \
    --test  samples/llama3/samples_orig_mmlu_all_test.npz \
    --arch cnn --kernel_size 7 --dropout 0.5 \
    --epochs 20 --batch 256 --lr 1e-4 \
    --save_dir models/llama3_pca128_mmlu_cnn_k7
```

### Training (combined — multiple npz merged in memory)
```bash
python classifier_pca_cnn.py \
    --model llama3 \
    --train samples/llama3/samples_orig_mmlu_all_train.npz \
            samples/llama3/samples_orig_mmlupro_all_train.npz \
    --test  samples/llama3/samples_orig_mmlu_all_test.npz \
            samples/llama3/samples_orig_mmlupro_all_test.npz \
    --arch cnn --kernel_size 7 --dropout 0.5 \
    --epochs 20 --batch 256 --lr 1e-4 \
    --save_dir models/llama3_pca128_mmlu_mmlupro_cnn_k7
```

> Multiple `--train` / `--test` files are merged **in memory** — no intermediate merged file is written to disk.

### Saved artifacts
```
models/llama3_pca128_mmlu_cnn_k7/
  model.pt          — model weights + config (arch, num_layers, pca_dim, ...)
  preprocessor.pkl  — per-layer StandardScaler + PCA objects
```

### Current best results (llama3-8B)

| Classifier | Trained on | Test AUC | Notes |
|---|---|---|---|
| `llama3_pca128_mmlu_cnn_k7` | MMLU | **0.812** | In-domain best |
| `llama3_pca128_mmlu_mmlupro_cnn_k7` | MMLU + MMLU-Pro | 0.789 | More diverse training |
| `llama3_pca128` | non-MMLU (GPQA/FACTOR/…) | 0.786 | Old architecture (v1, no residual) |

---

## Step 5 — Evaluate OOD Generalization

**Repo**: ConfSteer  
**Purpose**: Test the trained classifier on held-out benchmarks (different from training domain) to assess transferability.

```bash
# In: /data1/paveen/ConfSteer

# Evaluate on MMLU-Pro
python eval_classifier_ood.py \
    --benchmark mmlupro \
    --model llama3 --size 8B \
    --clf_dir models/llama3_pca128_mmlu_cnn_k7 \
    --role neutral

# Evaluate on GPQA
python eval_classifier_ood.py \
    --benchmark gpqa \
    --model llama3 --size 8B \
    --clf_dir models/llama3_pca128_mmlu_cnn_k7 \
    --role neutral

# Evaluate on MMLU (reverse OOD — using non-MMLU trained classifier)
python eval_classifier_ood.py \
    --benchmark mmlu \
    --model llama3 --size 8B \
    --clf_dir models/llama3_pca128 \
    --role neutral
```

**Supported benchmarks**: `mmlu`, `mmlupro`, `gpqa`

### OOD Results Summary

| Classifier | Trained on | Eval on | AUC | Mean per-task AUC |
|---|---|---|---|---|
| `mmlu_cnn_k7` | MMLU | MMLU-Pro | 0.732 | 0.659 |
| `mmlu_cnn_k7` | MMLU | GPQA | 0.574 | 0.565 |
| `mmlu_mmlupro_cnn_k7` | MMLU + MMLU-Pro | GPQA | 0.555 | 0.560 |
| `pca128` | non-MMLU | MMLU | 0.773 | 0.739 |

> GPQA remains a hard OOD target regardless of training diversity — graduate-level reasoning tasks have a fundamentally different hidden state geometry from MMLU-style tasks.

---

## Step 6 — Classifier-Guided Steering Benchmark

**Repo**: RolePlaying  
**Purpose**: Three-way comparison on actual benchmarks: no_steer vs always_steer vs classifier-guided steer. The classifier predicts which samples are currently wrong; only those get steered.

```bash
# In: /data1/paveen/RolePlaying
bash run_benchmark_clf_llama3.sh
```

**Script**: calls `get_answer_classifier_mmlupro.py` (MMLU-Pro) or `get_answer_classifier_mmlu.py` (MMLU)

**Key args** (set inside the shell script):
| Arg | Value | Description |
|---|---|---|
| `--clf_dir` | `ConfSteer/models/llama3_pca128_mmlu_cnn_k7` | Which classifier to use |
| `--configs` | `4-11-20` | alpha-start_layer-end_layer |
| `--mask_type` | `nmd` | Neuron masking method |
| `--percentage` | `0.5` | Top-50% neurons masked |
| `--roles` | `neutral` | Role for classifier evaluation |

**Output**:
```
RolePlaying/components/llama3/answer_clf_mmlupro/{task}_8B_answers.json
```

Each JSON contains three keys per sample: `answer_no_steer`, `answer_always_steer`, `answer_classifier`.

### Three-way results (llama3-8B, MMLU-Pro, α=4, layers 11–20)

| Condition | Accuracy | Steer Rate |
|---|---|---|
| no_steer | 35.74% | 0% |
| always_steer | 37.43% | 100% |
| classifier | **37.20%** | 69.4% |

> Classifier achieves near-`always_steer` accuracy with only 69.4% of interventions — reducing unnecessary steering by ~30%.

---

## Directory Structure Reference

```
/data1/paveen/
├── RolePlaying/
│   ├── get_answer_logits.py               # Step 1: MMLU orig answers + HS
│   ├── get_answer_logits_mmlupro.py       # Step 1: MMLU-Pro orig answers + HS
│   ├── get_answer_regenerate_logits.py    # Step 2: MMLU mdf answers
│   ├── get_answer_regenerate_logits_mmlupro.py  # Step 2: MMLU-Pro mdf answers
│   ├── get_answer_classifier_mmlu.py      # Step 6: classifier benchmark (MMLU)
│   ├── get_answer_classifier_mmlupro.py   # Step 6: classifier benchmark (MMLU-Pro)
│   ├── run_hidden_mmlu.sh                 # Step 1: batch HS extraction (MMLU)
│   ├── run_benchmark_llama3.sh            # Step 2: batch mdf generation
│   ├── run_benchmark_clf_llama3.sh        # Step 6: classifier benchmark
│   ├── components/llama3/
│   │   ├── answer_hs_mmlu/                # Step 1 output: orig answers (MMLU)
│   │   ├── answer_mmlupro/orig/           # Step 1 output: orig answers (MMLU-Pro)
│   │   ├── answer_mdf_mmlu/mdf_4/         # Step 2 output: steered answers
│   │   └── answer_clf_mmlupro/            # Step 6 output: classifier benchmark
│   └── benchmark/
│       ├── mmlupro_test.json
│       ├── gpqa_train.json
│       └── ...
│
└── ConfSteer/
    ├── prepare_samples_mmlu.py            # Step 3: MMLU → .npz
    ├── prepare_samples_mmlupro.py         # Step 3: MMLU-Pro → .npz
    ├── classifier_pca_cnn.py              # Step 4: train classifier
    ├── eval_classifier_ood.py             # Step 5: OOD evaluation
    ├── HiddenStates/llama3/
    │   ├── mmlu/{role}_{task}_8B.h5       # Step 1 output: HS files
    │   └── mmlupro/{role_prefix}_{task}_8B.h5
    ├── samples/llama3/
    │   ├── samples_orig_mmlu_all_train.npz  # Step 3 output
    │   └── samples_orig_mmlupro_all_train.npz
    └── models/
        └── llama3_pca128_mmlu_cnn_k7/     # Step 4 output
            ├── model.pt
            └── preprocessor.pkl
```

---

## Environment Setup

```bash
conda activate roleplaying   # shared env for both repos
# or
bash ConfSteer/setup_env.sh
```

Dependencies: `torch`, `transformers`, `h5py`, `numpy`, `scikit-learn`, `tqdm`

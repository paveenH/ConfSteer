# ConfSteer

## Project Overview

**ConfSteer** investigates whether a model's internal hidden states can predict
how it responds to confidence-directed activation steering — and therefore guide
*when* and *in which direction* to steer at inference time.

### Background

We use **RSN (Role-Shifted Network)** diff vectors injected into the model's
residual stream via forward hooks. Sweeping the steering coefficient alpha
across {-4, +4} produces two perturbed answer sets per sample. By comparing
these answers to the original (unsteered) answer and to the ground truth, we
assign a label to every sample that encodes whether steering improved, degraded,
or left unchanged the model's response.

A classifier trained on the **original hidden states** (before any steering)
should learn to predict which action to take at inference time:

| Predicted state | Action |
|---|---|
| Model is likely **wrong** | Apply **+4** steering → boost confidence |
| Model is likely **correct** | Apply **-4** steering or leave unchanged |
| Uncertain | No steering (class 0) |

### Why this matters

`orig_correct` is unknown at inference time — the classifier must recover it
from hidden states alone. This is equivalent to **correctness detection from
hidden representations**, using steering response as a proxy label. Prior work
(logit lens, probing classifiers) confirms that correctness information is
encoded in intermediate layers, making this task feasible.

---

## Prompt Format

All answers (original and steered) are collected using the **default-E suite** from
`RolePlaying/template.py`. The prompt includes an extra option **E) I am not sure.**
to capture model uncertainty.

### Role-based template (`neutral` excluded)

```
Would you answer the following question with A, B, C, D or E?
Question: {question}
A) ...  B) ...  C) ...  D) ...
E) I am not sure.
Now you are an honest {character}, your answer among "A, B, C, D, E" is:
```

`{character}` is filled by the role name (e.g. `confident`, `expert`, `biology_expert`).

### Neutral template (no role injection)

```
Would you answer the following question with A, B, C, D or E?
Question: {question}
A) ...  B) ...  C) ...  D) ...
E) I am not sure.
Your answer among "A, B, C, D, E" is:
```

The `neutral` role uses a separate template with **no role phrase** at all — it serves
as the no-intervention baseline.

### Roles used

| Role | Template phrase | Notes |
|---|---|---|
| `neutral` | *(none)* | Baseline; no character injection |
| `confident` | `Now you are an honest confident` | |
| `unconfident` | `Now you are an honest unconfident` | |
| `expert` | `Now you are an honest expert` | Generic tasks |
| `non_expert` | `Now you are an honest non_expert` | |
| `student` | `Now you are an honest student` | |
| `person` | `Now you are an honest person` | |
| `{subject}_expert` | `Now you are an honest {subject}_expert` | mmlupro only (e.g. `biology_expert`) |
| `{subject}_non_expert` | `Now you are an honest {subject}_non_expert` | mmlupro only |
| `{subject}_student` | `Now you are an honest {subject}_student` | mmlupro only |

The hidden state at the **last token** of this prompt is extracted and saved to `.h5`.
Steered answers (`mdf_4`, `mdf_-4`) use the same prompt with the RSN diff vector injected
into the residual stream at the specified layers during the forward pass — the prompt
itself does not change.

---

## Models

| Model | Size | Architecture |
|---|---|---|
| `llama3` | 8B | LLaMA-3-8B-Instruct |
| `qwen3` | 8B | Qwen3-8B |

Both models are loaded in `bfloat16`. Hidden states have shape `(N, 33, 4096)` for llama3
and `(N, 37, 4096)` for qwen3 (33/37 layers including embedding, 4096 hidden dim).

---

## End-to-End Workflow

```
Step 1  Collect answers (RolePlaying repo)
        get_answer_logits_*.py --save
        → answer/{model}/{task}/orig/       original answers + hidden states (.h5)
        → answer/{model}/{task}/mdf_4/      answers after +4 steering
        → answer/{model}/{task}/mdf_-4/     answers after -4 steering

Step 2  Generate labels
        python make_labels.py --model llama3 --params 20_11_20
        → labels/{model}/{task}/labels_*.json

Step 3  Pre-extract hidden states
        python prepare_samples.py --model llama3 --ratio 1.0
        → samples/{model}/samples_binary_all_train.npz
        → samples/{model}/samples_binary_all_test.npz
        → samples/{model}/samples_three_all_train.npz
        → samples/{model}/samples_three_all_test.npz

Step 4  Train classifier
        python classifier_binary.py --model llama3 --layer 25 \
          --train samples/llama3/samples_binary_all_train.npz \
          --test  samples/llama3/samples_binary_all_test.npz
        python classifier_cnn.py --model llama3 \
          --train samples/llama3/samples_binary_all_train.npz \
          --test  samples/llama3/samples_binary_all_test.npz
        → plots/
```

> **Data note:** Steps 1–3 are run on the server (`/data1/paveen/ConfSteer/`).
> The `answer/` and `HiddenStates/` directories are not committed to git due to size.
> Pre-extracted `samples/*.npz` files are the recommended starting point for classifier development.

---

## Repository Structure

```
ConfSteer/
├── answer/                  # Model answers (not in git; server: /data1/paveen/ConfSteer/answer)
│   ├── llama3/
│   │   ├── {task}/orig/     original answers (.json)
│   │   ├── {task}/mdf_4/    +4 steered answers (.json)
│   │   └── {task}/mdf_-4/   -4 steered answers (.json)
│   └── qwen3/
├── labels/                  # Per-sample steering labels (generated by make_labels.py)
│   ├── llama3/
│   │   ├── mmlupro/         labels_*.json
│   │   ├── factor/
│   │   ├── tqa/
│   │   └── ...
│   └── qwen3/
│       └── ...
├── HiddenStates/            # Per-role H5 hidden states (server: /data1/paveen/ConfSteer/HiddenStates)
│   ├── llama3/
│   │   ├── mmlupro/         {role}_{topic}_8B.h5
│   │   ├── truthfulqa/      {role}_TruthfulQA_MC{1,2}_llama3_8B.h5
│   │   └── ...
│   └── qwen3/
│       └── ...
├── samples/                 # Pre-extracted numpy arrays (generated by prepare_samples.py)
│   ├── llama3/
│   │   ├── samples_binary_all_train.npz   # question-level train split, downsampled
│   │   ├── samples_binary_all_test.npz    # question-level test split, original distribution
│   │   ├── samples_three_all_train.npz
│   │   └── samples_three_all_test.npz
│   └── qwen3/
│       ├── samples_binary_all_train.npz
│       ├── samples_binary_all_test.npz
│       ├── samples_three_all_train.npz
│       └── samples_three_all_test.npz
├── make_labels.py           # Step 2: generate label files from answer comparisons
├── prepare_samples.py       # Step 3: pre-extract hidden states from H5 → npz
├── classifier_binary.py     # Step 4: binary classifier (should we apply +4 steering?)
├── run_classifier_binary.sh # Shell script to run binary classifier
├── RESULT.md                # Classification result log
├── setup_env.sh             # Conda environment setup
└── README.md
```

---

## Hidden States

Hidden states are extracted during the **original (unsteered) forward pass**
using `get_answer_logits_*.py` with `--save`. Only the **last token** hidden
state is stored, across all layers. **Each role has its own H5 file.**

### Storage format

```
HiddenStates/{model}/{task}/{role}_{stem}_{size}.h5
```

Examples:
```
HiddenStates/llama3/mmlupro/neutral_astronomy_8B.h5
HiddenStates/llama3/mmlupro/confident_astronomy_8B.h5
HiddenStates/llama3/truthfulqa/neutral_TruthfulQA_MC1_llama3_8B.h5
HiddenStates/qwen3/gpqa/expert_GPQA_(gpqa_main)_8B.h5
```

For mmlupro, task-specific roles use a `{subject}_{base_role}` pattern:
```
HiddenStates/llama3/mmlupro/biology_expert_Biology_8B.h5
```

### H5 structure

```python
import h5py
with h5py.File("neutral_astronomy_8B.h5", "r") as f:
    hs = f["hidden_states"]   # shape: (N, n_layers, hidden_dim)
                              # e.g.  (80, 33, 4096) for llama3-8B
```

Rows are ordered identically to the original answer JSON — alignment is by
**integer index**, no text matching required.

---

## Label Files

Labels are generated by `make_labels.py` by comparing three answer files per task:

```
{model}/{task}/orig/          original answers (no steering)
{model}/{task}/mdf_4/         answers after +4 steering
{model}/{task}/mdf_-4/        answers after -4 steering
```

Output: `labels/{model}/{task}/labels_{orig_filename}.json`

### Per-sample fields

| Field | Type | Description |
|---|---|---|
| `index` | int | Row index — maps directly to `h5["hidden_states"][index]` |
| `role` | str | Role name (e.g. `neutral`, `confident`, `biology_expert`) |
| `text` | str | Question + choices (for alignment verification) |
| `true_label` | str / list | Ground truth answer letter(s) |
| `answer_orig` | str | Model answer without steering |
| `answer_pos4` | str | Model answer after +4 steering |
| `answer_neg4` | str | Model answer after -4 steering |
| `orig_correct` | bool | Whether the original answer was correct |
| `label_pos4` | int | +4 steering outcome (see encoding below) |
| `label_neg4` | int | -4 steering outcome (see encoding below) |

### Raw label encoding

```
+4 steering (confidence boost):
  +1  = worked    — wrong → correct
  -1  = backfire  — correct → wrong
   0  = no change

-4 steering (confidence suppression):
  +1  = backfire  — wrong → correct
  -1  = worked    — correct → wrong
   0  = no change
```

---

## Sample Preparation (`prepare_samples.py`)

Because the H5 files are large and scattered, `prepare_samples.py` pre-extracts
all hidden states into two compact `.npz` files — one per classifier formulation.
This avoids repeated H5 I/O at training time.

### Label encoding in npz

| npz `y` value | Meaning |
|---|---|
| **Binary** `y=1` | `label_pos4 == +1` — +4 steering works (wrong→correct) |
| **Binary** `y=0` | All other outcomes |
| **Three-class** `y=1` | `label_pos4 == +1` — pos4 works |
| **Three-class** `y=2` | `label_pos4 == -1` — neg4 works (backfire of +4) |
| **Three-class** `y=0` | No change |

### Split strategy

Samples are split at the **question level**: all roles of the same question
`(task, orig_stem, index)` go to the same split, preventing data leakage
across roles.

- **Train split**: class 0 downsampled to `ratio × n(y=1)` (balanced)
- **Test split**: original class distribution (no downsampling)

| File | Class 0 definition | Train downsampled | Test |
|---|---|---|---|
| `samples_binary_*_train/test.npz` | no_change + neg4_works | `ratio × n(y=1)` | original distribution |
| `samples_three_*_train/test.npz` | no_change only | `ratio × max(n(y=1), n(y=2))` | original distribution |

### Usage

```bash
python prepare_samples.py --model llama3
python prepare_samples.py --model qwen3 --ratio 1.0
python prepare_samples.py --model llama3 --roles neutral confident expert
```

Output (server: `/data1/paveen/ConfSteer/samples/`):
```
samples/{model}/samples_binary_{roles}_train.npz   # X: float16, y: int8
samples/{model}/samples_binary_{roles}_test.npz    # X: float16, y: int8
samples/{model}/samples_three_{roles}_train.npz    # X: float16, y: int8
samples/{model}/samples_three_{roles}_test.npz     # X: float16, y: int8
```

### npz contents

```python
import numpy as np
data = np.load("samples_binary_all_train.npz", allow_pickle=False)
X     = data["X"]      # (N, n_layers, hidden_dim)  float16
y     = data["y"]      # (N,)                        int8
meta  = data["meta"]   # (N,) JSON strings — {task, orig_stem, role, index}
roles = data["roles"]  # role list used
```

---

## Binary Classifier (`classifier_binary.py`)

Trains a Logistic Regression to predict: **should we apply +4 steering?**

### Usage

```bash
# Recommended: question-level train/test split
python classifier_binary.py --model llama3 --layer 25 \
  --train samples/llama3/samples_binary_all_train.npz \
  --test  samples/llama3/samples_binary_all_test.npz

# With PCA or layer sweep
python classifier_binary.py --model llama3 --layer 25 --pca 50 \
  --train samples/llama3/samples_binary_all_train.npz \
  --test  samples/llama3/samples_binary_all_test.npz
python classifier_binary.py --model llama3 --layer_sweep \
  --train samples/llama3/samples_binary_all_train.npz \
  --test  samples/llama3/samples_binary_all_test.npz

# Legacy: single npz with sample-level split (deprecated)
python classifier_binary.py --model llama3 --layer 25 --samples samples/llama3/samples_binary_all.npz
```

### Pipeline

```
[1] Load _train.npz / _test.npz  (already downsampled, float16)
[2] Cast X to float32  →  slice layer  →  (N, hidden_dim)
[3] StandardScaler fit on train, transform test
[4] LogisticRegression (class_weight="balanced"), 5-fold CV on train
[5] Evaluate on held-out test: F1, ROC-AUC, classification_report, confusion matrix
```

---

## CNN Classifier (`classifier_cnn.py`)

Trains a 1D-CNN with layer attention over all layers (or a layer range) to predict:
**should we apply +4 steering?**

Architecture: Linear proj (D→proj_dim) → 1D-CNN (kernel=3) → Layer Attention → MLP → 2 classes

### Usage

```bash
# Recommended: question-level train/test split
python classifier_cnn.py --model llama3 \
  --train samples/llama3/samples_binary_all_train.npz \
  --test  samples/llama3/samples_binary_all_test.npz

# With custom hyperparameters
python classifier_cnn.py --model llama3 \
  --train samples/llama3/samples_binary_all_train.npz \
  --test  samples/llama3/samples_binary_all_test.npz \
  --layers 10-25 --proj_dim 128 --cnn_channels 256 --epochs 30

# Legacy: single npz with sample-level split (deprecated)
python classifier_cnn.py --model llama3 --samples samples/llama3/samples_binary_all.npz
```

### Pipeline

```
[1] Load _train.npz / _test.npz  (already downsampled, float16)
[2] Select layer range  →  (N, L, hidden_dim)
[3] Per-layer StandardScaler fit on train, transform test
[4] LayerAttentionCNN: proj → 1D-CNN → attention → MLP
[5] Train with AdamW + CosineAnnealingLR; best checkpoint by val loss
[6] Evaluate on held-out test: Acc, ROC-AUC, classification_report
```

---

## Tasks Covered

| Task | Files | Notes |
|---|---|---|
| `mmlupro` | 60+ subject files | MMLU-Pro format; task-specific roles (e.g. biology_expert) |
| `factor` | Expert, News, Wiki (MC4) | |
| `arlsat` | dev, test, train | AR-LSAT |
| `gpqa` | diamond, extended, main | |
| `logiqa` | MRC-test | |
| `tqa_mc1` | TruthfulQA MC1 | |
| `tqa_mc2` | TruthfulQA MC2 | |

---

## Environment Setup

```bash
bash setup_env.sh        # creates conda env "confsteer"
conda activate confsteer
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `h5py`, `torch`, `pandas`, `tqdm`

---

## Results

See [RESULT.md](RESULT.md) for the full classification result log.

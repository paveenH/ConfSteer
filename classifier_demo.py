"""
ConfSteer Classifier Demo
=========================
Goal: Given hidden states at the last token (middle layer), predict whether
      +4 steering (+1), -4 steering (-1), or no steering (0) is optimal.

Label definition:
  +1  : label_pos4 == +1  (wrong→correct under +4)  [includes conflict cases]
  -1  : label_neg4 == +1  (wrong→correct under -4)
   0  : all others (no improvement, or degradation) — downsampled

Usage:
  # Real run on server
  python classifier_demo.py --model llama3

  # Local dry-run with mock data (no h5 files needed)
  python classifier_demo.py --model llama3 --mock
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent

LABEL_DIR   = BASE_DIR / "labels"
HIDDEN_DIR  = BASE_DIR / "HiddenStates"   # server path: /data1/paveen/ConfSteer/HiddenStates

# ==================== Config ====================
MIDDLE_LAYER = 16        # target layer index (0-based); adjust per model
RANDOM_SEED  = 42
NO_CHANGE_RATIO = 1.0    # no_change samples = ratio × max(pos, neg) count


# ==================== H5 filename mapping ====================

def get_h5_path(model: str, task: str, orig_stem: str) -> Path:
    """
    Map label file stem → h5 file path.

    Examples:
      astronomy_8B_answers      → neutral_astronomy_8B.h5
      GPQA_(gpqa_diamond)_8B_answers → neutral_GPQA_(gpqa_diamond)_8B.h5
      TruthfulQA_MC1_llama3_8B_mc1   → neutral_TruthfulQA_MC1_mc1_8B.h5  (tqa special)
    """
    # Strip trailing _answers or _answers_* suffix
    stem = orig_stem
    for suffix in ["_answers", "_mc1", "_mc2"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    # TQA: rename task folder tqa_mc1/tqa_mc2 → truthfulqa
    h5_task = "truthfulqa" if task.startswith("tqa") else task

    # Strip model name token if present (e.g. "llama3_8B" → "8B" handled by neutral_ prefix)
    h5_name = f"neutral_{stem}.h5"
    return HIDDEN_DIR / model / h5_task / h5_name


# ==================== Label loading ====================

def assign_class(lp: int, ln: int) -> int:
    """Map (label_pos4, label_neg4) → class {+1, -1, 0}."""
    if lp == 1:        # +4 works (includes conflict lp==ln==1)
        return 1
    if ln == 1:        # -4 works (wrong→correct)
        return -1
    return 0           # no improvement


def load_labels(model: str):
    """Return list of (label_json_path, task, orig_stem, sample_list)."""
    result = []
    pattern = str(LABEL_DIR / model / "**" / "labels_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        p = Path(path)
        task = p.parent.name
        orig_stem = p.stem[len("labels_"):]   # strip "labels_" prefix
        with open(p) as f:
            data = json.load(f)["data"]
        result.append((p, task, orig_stem, data))
    return result


# ==================== Feature extraction ====================

def extract_features_real(model: str, label_files, layer: int = MIDDLE_LAYER):
    """Load hidden states from h5 and extract middle-layer features."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for real mode: pip install h5py")

    X, y = [], []
    skipped = 0

    for label_path, task, orig_stem, samples in label_files:
        h5_path = get_h5_path(model, task, orig_stem)
        if not h5_path.exists():
            print(f"  [skip] h5 not found: {h5_path}")
            skipped += 1
            continue

        with h5py.File(h5_path, "r") as hf:
            hs = hf["hidden_states"]   # shape: (N, layers, hidden_dim)
            n_layers = hs.shape[1]
            l = min(layer, n_layers - 1)

            for s in samples:
                cls = assign_class(s["label_pos4"], s["label_neg4"])
                idx = s["index"]
                feat = hs[idx, l, :]   # (hidden_dim,)
                X.append(feat)
                y.append(cls)

    print(f"  Loaded {len(X)} samples ({skipped} files skipped)")
    return np.array(X, dtype=np.float32), np.array(y)


def extract_features_mock(model: str, label_files):
    """
    Mock feature extraction for local testing (no h5 needed).
    Simulates hidden_dim=4096, with class-specific Gaussian clusters.
    """
    print("  [MOCK] Generating synthetic features (hidden_dim=4096)")
    hidden_dim = 4096
    rng = np.random.default_rng(RANDOM_SEED)

    # Class centroids — small offset to make task learnable but hard
    centroids = {1: 0.05, -1: -0.05, 0: 0.0}

    X, y = [], []
    for _, _, _, samples in label_files:
        for s in samples:
            cls = assign_class(s["label_pos4"], s["label_neg4"])
            feat = rng.normal(centroids[cls], 1.0, hidden_dim).astype(np.float32)
            X.append(feat)
            y.append(cls)

    return np.array(X), np.array(y)


# ==================== Sampling ====================

def balanced_sample(X, y, ratio=NO_CHANGE_RATIO, seed=RANDOM_SEED):
    """
    Downsample class 0 to ratio × max(|class +1|, |class -1|).
    Classes +1 and -1 are kept in full.
    """
    rng = random.Random(seed)

    idx_pos  = np.where(y ==  1)[0].tolist()
    idx_neg  = np.where(y == -1)[0].tolist()
    idx_zero = np.where(y ==  0)[0].tolist()

    n_keep = int(ratio * max(len(idx_pos), len(idx_neg)))
    idx_zero_sampled = rng.sample(idx_zero, min(n_keep, len(idx_zero)))

    keep = idx_pos + idx_neg + idx_zero_sampled
    rng.shuffle(keep)
    keep = np.array(keep)
    return X[keep], y[keep]


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--layer", type=int, default=MIDDLE_LAYER,
                        help="Hidden state layer index (0-based)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock features (no h5 files needed)")
    args = parser.parse_args()

    layer = args.layer

    print(f"\n{'='*50}")
    print(f"  ConfSteer Classifier Demo")
    print(f"  Model : {args.model}")
    print(f"  Layer : {layer}")
    print(f"  Mode  : {'MOCK' if args.mock else 'REAL'}")
    print(f"{'='*50}\n")

    # Load labels
    print("[1] Loading labels...")
    label_files = load_labels(args.model)
    print(f"  Found {len(label_files)} label files")

    # Extract features
    print("\n[2] Extracting features...")
    if args.mock:
        X, y = extract_features_mock(args.model, label_files)
    else:
        X, y = extract_features_real(args.model, label_files, layer)

    print(f"  Raw class dist — +1: {(y==1).sum()}, -1: {(y==-1).sum()}, 0: {(y==0).sum()}")

    # Downsample class 0
    print("\n[3] Balancing classes...")
    X, y = balanced_sample(X, y)
    print(f"  Balanced — +1: {(y==1).sum()}, -1: {(y==-1).sum()}, 0: {(y==0).sum()}")
    print(f"  Total samples: {len(y)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Standardize
    print("\n[4] Training Logistic Regression...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    print("\n[5] Evaluation on test set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["neg(-1)", "no_change(0)", "pos(+1)"]))

    print("Confusion matrix (rows=true, cols=pred):")
    print("  labels: -1, 0, +1")
    print(confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]))


if __name__ == "__main__":
    main()

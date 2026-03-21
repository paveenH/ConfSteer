"""
ConfSteer Binary Classifier
============================
Binary classification: should we apply +4 steering?

Label definition:
  +1 (steer)    : label_pos4 == +1  (wrong→correct under +4 steering)
   0 (no_steer) : all others        (already correct, or steering doesn't help)

Class 0 is downsampled to ratio × count(+1).

Usage:
  python classifier_binary.py --model llama3 --layer 19
  python classifier_binary.py --model llama3 --layer 19 --pca 50
  python classifier_binary.py --model llama3 --layer 19 --pca 50 --visualize
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ==================== Paths ====================
BASE_DIR   = Path(__file__).parent
LABEL_DIR  = BASE_DIR / "labels"
HIDDEN_DIR = BASE_DIR / "HiddenStates"

# ==================== Config ====================
MIDDLE_LAYER    = 19
RANDOM_SEED     = 42
NO_CHANGE_RATIO = 1.0


# ==================== H5 path mapping ====================

def get_h5_path(model: str, task: str, orig_stem: str) -> Path:
    stem = orig_stem
    for suffix in ["_answers", "_mc1", "_mc2"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    h5_task = "truthfulqa" if task.startswith("tqa") else task
    return HIDDEN_DIR / model / h5_task / f"neutral_{stem}.h5"


# ==================== Label loading ====================

def assign_binary(lp: int) -> int:
    """
    +1 : +4 steering worked (wrong→correct)
     0 : everything else
    """
    return 1 if lp == 1 else 0


def load_labels(model: str):
    result = []
    pattern = str(LABEL_DIR / model / "**" / "labels_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        p = Path(path)
        task = p.parent.name
        orig_stem = p.stem[len("labels_"):]
        with open(p) as f:
            data = json.load(f)["data"]
        result.append((p, task, orig_stem, data))
    return result


# ==================== Feature extraction ====================

def extract_features(model: str, label_files, layer: int):
    import h5py
    X, y = [], []
    skipped = 0
    for label_path, task, orig_stem, samples in label_files:
        h5_path = get_h5_path(model, task, orig_stem)
        if not h5_path.exists():
            print(f"  [skip] h5 not found: {h5_path}")
            skipped += 1
            continue
        with h5py.File(h5_path, "r") as hf:
            hs = hf["hidden_states"]
            n_layers = hs.shape[1]
            l = min(layer, n_layers - 1)
            for s in samples:
                cls = assign_binary(s["label_pos4"])
                X.append(hs[s["index"], l, :])
                y.append(cls)
    print(f"  Loaded {len(X)} samples ({skipped} files skipped)")
    return np.array(X, dtype=np.float32), np.array(y)


# ==================== Sampling ====================

def balanced_sample(X, y, ratio=NO_CHANGE_RATIO, seed=RANDOM_SEED):
    rng = random.Random(seed)
    idx_pos  = np.where(y == 1)[0].tolist()
    idx_zero = np.where(y == 0)[0].tolist()
    n_keep = int(ratio * len(idx_pos))
    idx_zero_sampled = rng.sample(idx_zero, min(n_keep, len(idx_zero)))
    keep = np.array(idx_pos + idx_zero_sampled)
    rng.shuffle(keep.tolist())
    return X[keep], y[keep]


# ==================== Visualization ====================

def visualize(X_scaled, y, model, layer, pca_n, out_dir):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "#95a5a6", 1: "#e74c3c"}
    labels_map = {0: "no_steer(0)", 1: "steer(+1)"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Binary Separability — {model}  layer={layer}  pca={pca_n}", fontsize=13)

    pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca2.fit_transform(X_scaled)
    ax = axes[0]
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[cls], label=labels_map[cls], alpha=0.4, s=10)
    ax.set_title(f"PCA 2D  (var: {pca2.explained_variance_ratio_.sum()*100:.1f}%)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=2)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y).squeeze()
    ax = axes[1]
    for cls in [0, 1]:
        mask = y == cls
        ax.hist(X_lda[mask], bins=40, alpha=0.6, color=colors[cls], label=labels_map[cls])
    ax.set_title("LDA 1D distribution")
    ax.set_xlabel("LD1"); ax.legend()

    plt.tight_layout()
    out_path = out_dir / f"binary_{model}_layer{layer}_pca{pca_n}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--layer",  type=int, default=MIDDLE_LAYER)
    parser.add_argument("--pca",    type=int, default=0,
                        help="PCA components (0=disabled)")
    parser.add_argument("--ratio",  type=float, default=NO_CHANGE_RATIO,
                        help="class-0 downsample ratio (default: 1.0)")
    parser.add_argument("--C",      type=float, default=1.0,
                        help="LR regularization strength")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  ConfSteer Binary Classifier")
    print(f"  Model : {args.model}  Layer : {args.layer}")
    print(f"  PCA   : {args.pca if args.pca > 0 else 'disabled'}")
    print(f"  Ratio : {args.ratio}  C : {args.C}")
    print(f"{'='*50}\n")

    # ── Load ──
    print("[1] Loading labels...")
    label_files = load_labels(args.model)
    print(f"  Found {len(label_files)} label files")

    print("\n[2] Extracting features...")
    X, y = extract_features(args.model, label_files, args.layer)
    print(f"  Raw — steer(+1): {(y==1).sum()}, no_steer(0): {(y==0).sum()}")

    # ── Balance ──
    print("\n[3] Balancing...")
    X, y = balanced_sample(X, y, ratio=args.ratio)
    print(f"  Balanced — steer(+1): {(y==1).sum()}, no_steer(0): {(y==0).sum()}")
    print(f"  Total: {len(y)}")

    # ── Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA ──
    pca_n = args.pca
    if pca_n > 0:
        pca = PCA(n_components=pca_n, random_state=RANDOM_SEED)
        X_scaled = pca.fit_transform(X_scaled)
        var = pca.explained_variance_ratio_.sum() * 100
        print(f"\n  PCA({pca_n}): explained variance = {var:.1f}%")

    # ── Visualize ──
    if args.visualize:
        print("\n[viz] Generating plots...")
        visualize(X_scaled, y, args.model, args.layer, pca_n,
                  BASE_DIR / "plots")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # ── Logistic Regression ──
    print("\n[4] Training Logistic Regression (class_weight=balanced)...")
    clf = LogisticRegression(
        max_iter=1000, C=args.C,
        class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 5-fold CV ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="f1")
    print(f"  5-fold CV F1 (steer class): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Evaluate ──
    print("\n[5] Evaluation on test set:")
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred,
                                 target_names=["no_steer(0)", "steer(+1)"]))
    print(f"  ROC-AUC: {auc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("  labels: 0, +1")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()

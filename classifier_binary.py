"""
ConfSteer Binary Classifier
============================
Binary classification: should we apply +4 steering?

Label definition:
  +1 (steer)    : label_pos4 == +1  (wrong→correct under +4 steering)
   0 (no_steer) : all others        (already correct, or steering doesn't help)

Usage (question-level split, recommended):
  python classifier_binary.py --model llama3 --layer 25 \
    --train samples/llama3/samples_binary_all_train.npz \
    --test  samples/llama3/samples_binary_all_test.npz

  python classifier_binary.py --model llama3 --layer_sweep --pca 50 \
    --train samples/llama3/samples_binary_all_train.npz \
    --test  samples/llama3/samples_binary_all_test.npz

Usage (legacy: single npz, sample-level split):
  python classifier_binary.py --model llama3 --layer 25 \
    --samples samples/llama3/samples_binary_all.npz
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
SAMPLE_DIR = BASE_DIR / "samples"

# ==================== Config ====================
MIDDLE_LAYER    = 19
RANDOM_SEED     = 42
NO_CHANGE_RATIO = 1.0


# ==================== Sample loading (from prepare_samples.py output) ====================

def load_samples(path: Path):
    """Load pre-extracted samples from .npz file (output of prepare_samples.py)."""
    data = np.load(path, allow_pickle=False)
    X     = data["X"]      # (N, n_layers, hidden_dim)
    y     = data["y"]      # (N,) int8
    roles = list(data["roles"])
    print(f"  Loaded samples: shape={X.shape}, roles={roles}")
    print(f"  y — steer(1): {(y==1).sum()}, no_steer(0): {(y==0).sum()}")
    return X, y.astype(np.int64)


# ==================== H5 path mapping (legacy) ====================

def get_h5_path(model: str, task: str, orig_stem: str, role: str = "neutral") -> Path:
    stem = orig_stem
    for suffix in ["_answers", "_mc1", "_mc2"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    h5_task = "truthfulqa" if task.startswith("tqa") else task
    return HIDDEN_DIR / model / h5_task / f"{role}_{stem}.h5"


# ==================== Label loading (legacy) ====================

def assign_binary(lp: int) -> int:
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


# ==================== Feature extraction (legacy, reads H5 directly) ====================

def extract_features(model: str, label_files, layer: int):
    """Load a single layer from H5. Returns (N, hidden_dim)."""
    import h5py
    X, y = [], []
    skipped_paths = set()
    for _, task, orig_stem, samples in label_files:
        # Group by role so each H5 is opened once
        by_role = {}
        for s in samples:
            by_role.setdefault(s.get("role", "neutral"), []).append(s)
        for role, role_samples in by_role.items():
            h5_path = get_h5_path(model, task, orig_stem, role)
            if not h5_path.exists():
                if h5_path not in skipped_paths:
                    print(f"  [skip] h5 not found: {h5_path}")
                    skipped_paths.add(h5_path)
                continue
            with h5py.File(h5_path, "r") as hf:
                hs = hf["hidden_states"]
                l = min(layer, hs.shape[1] - 1)
                for s in role_samples:
                    X.append(hs[s["index"], l, :])
                    y.append(assign_binary(s["label_pos4"]))
    print(f"  Loaded {len(X)} samples ({len(skipped_paths)} H5 files skipped)")
    return np.array(X, dtype=np.float32), np.array(y)


def extract_all_layers(model: str, label_files):
    """Load all layers from H5. Returns (N, n_layers, hidden_dim)."""
    import h5py
    X, y = [], []
    skipped_paths = set()
    for _, task, orig_stem, samples in label_files:
        by_role = {}
        for s in samples:
            by_role.setdefault(s.get("role", "neutral"), []).append(s)
        for role, role_samples in by_role.items():
            h5_path = get_h5_path(model, task, orig_stem, role)
            if not h5_path.exists():
                skipped_paths.add(h5_path)
                continue
            with h5py.File(h5_path, "r") as hf:
                hs = hf["hidden_states"]
                for s in role_samples:
                    X.append(hs[s["index"], :, :])
                    y.append(assign_binary(s["label_pos4"]))
    X = np.array(X, dtype=np.float32)
    print(f"  Loaded {X.shape[0]} samples ({len(skipped_paths)} H5 files skipped), shape: {X.shape}")
    return X, np.array(y)


# ==================== Sampling ====================

def balanced_sample(X, y, ratio=NO_CHANGE_RATIO, seed=RANDOM_SEED):
    rng = random.Random(seed)
    idx_pos  = np.where(y == 1)[0].tolist()
    idx_zero = np.where(y == 0)[0].tolist()
    n_keep   = int(ratio * len(idx_pos))
    idx_zero_sampled = rng.sample(idx_zero, min(n_keep, len(idx_zero)))
    keep = np.array(idx_pos + idx_zero_sampled)
    return X[keep], y[keep]


def balanced_indices(y, ratio=NO_CHANGE_RATIO, seed=RANDOM_SEED):
    """Return balanced indices without slicing X (used when X is 3D)."""
    rng = random.Random(seed)
    idx_pos  = np.where(y == 1)[0].tolist()
    idx_zero = np.where(y == 0)[0].tolist()
    n_keep   = int(ratio * len(idx_pos))
    idx_zero_sampled = rng.sample(idx_zero, min(n_keep, len(idx_zero)))
    return np.array(idx_pos + idx_zero_sampled)


# ==================== Core: preprocess + train LR ====================

def run_lr(X, y, pca_n: int, C: float):
    """
    Scale → optional PCA → LogisticRegression with 5-fold CV.
    Returns: (clf, X_scaled, cv_f1_mean, cv_f1_std, cv_auc_mean)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if pca_n > 0:
        pca = PCA(n_components=min(pca_n, X_scaled.shape[1]), random_state=RANDOM_SEED)
        X_scaled = pca.fit_transform(X_scaled)
        var = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA({pca_n}): explained variance = {var:.1f}%")

    clf = LogisticRegression(
        max_iter=1000, C=C,
        class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    f1_scores  = cross_val_score(clf, X_scaled, y, cv=cv, scoring="f1")
    auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    clf.fit(X_scaled, y)   # fit on full data for downstream use
    return clf, X_scaled, f1_scores.mean(), f1_scores.std(), auc_scores.mean()


# ==================== Visualization ====================

def visualize(X_scaled, y, model, layer, pca_n, out_dir):
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


def plot_learning_curve(clf, X, y, model, layer, pca_n, out_dir):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(RANDOM_SEED)
    fracs = np.linspace(0.4, 1.0, 7)
    n = len(y)
    train_sizes_list, train_mean_list, train_std_list = [], [], []
    val_mean_list, val_std_list = [], []

    for frac in fracs:
        size = int(frac * n)
        fold_train_f1, fold_val_f1 = [], []
        cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for train_idx, val_idx in cv_obj.split(X, y):
            # subsample train_idx to `size` while keeping both classes
            idx0 = train_idx[y[train_idx] == 0]
            idx1 = train_idx[y[train_idx] == 1]
            n_per = size // 2
            s0 = idx0[rng.choice(len(idx0), min(n_per, len(idx0)), replace=False)]
            s1 = idx1[rng.choice(len(idx1), min(n_per, len(idx1)), replace=False)]
            sub = np.concatenate([s0, s1])
            if len(np.unique(y[sub])) < 2:
                continue
            est = LogisticRegression(max_iter=1000, C=clf.C,
                                     class_weight="balanced",
                                     random_state=RANDOM_SEED, n_jobs=-1)
            est.fit(X[sub], y[sub])
            fold_train_f1.append(f1_score(y[sub], est.predict(X[sub])))
            fold_val_f1.append(f1_score(y[val_idx], est.predict(X[val_idx])))

        train_sizes_list.append(size)
        train_mean_list.append(np.mean(fold_train_f1))
        train_std_list.append(np.std(fold_train_f1))
        val_mean_list.append(np.mean(fold_val_f1))
        val_std_list.append(np.std(fold_val_f1))

    train_sizes = np.array(train_sizes_list)
    train_mean  = np.array(train_mean_list)
    train_std   = np.array(train_std_list)
    val_mean    = np.array(val_mean_list)
    val_std     = np.array(val_std_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#2ecc71", label="Train F1")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2ecc71")
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", label="Val F1 (CV)")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#e74c3c")
    ax.set_title(f"Learning Curve — {model}  layer={layer}  pca={pca_n}")
    ax.set_xlabel("Training samples"); ax.set_ylabel("F1 (steer class)")
    ax.legend(); ax.grid(True); ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = out_dir / f"learning_curve_{model}_layer{layer}_pca{pca_n}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    gap = val_mean[-1] - val_mean[0]
    print(f"  Val F1: {val_mean[0]:.3f} (n={train_sizes[0]}) → {val_mean[-1]:.3f} (n={train_sizes[-1]})")
    if gap > 0.03:
        print(f"  → still rising (+{gap:.3f}): more data would help")
    else:
        print(f"  → plateaued ({gap:+.3f}): signal strength is the bottleneck")


def plot_layer_sweep_from_array(X_all, y_all, model: str, pca_n: int, ratio: float, C: float, out_dir: Path,
                                already_balanced: bool = False):
    import matplotlib.pyplot as plt

    n_layers = X_all.shape[1]

    if already_balanced:
        X_bal, y_bal = X_all, y_all
    else:
        keep  = balanced_indices(y_all, ratio=ratio)
        X_bal = X_all[keep]
        y_bal = y_all[keep]
    print(f"  Balanced — steer(+1): {(y_bal==1).sum()}, no_steer(0): {(y_bal==0).sum()}")

    cv_means, cv_stds, aucs = [], [], []
    print(f"  Sweeping {n_layers} layers (pca={pca_n})...")
    for layer in range(n_layers):
        X = X_bal[:, layer, :]
        _, _, f1_mean, f1_std, auc_mean = run_lr(X, y_bal, pca_n=pca_n, C=C)
        cv_means.append(f1_mean)
        cv_stds.append(f1_std)
        aucs.append(auc_mean)
        print(f"    layer {layer:2d}: F1={f1_mean:.3f}±{f1_std:.3f}  AUC={auc_mean:.3f}")

    cv_means = np.array(cv_means)
    cv_stds  = np.array(cv_stds)
    aucs     = np.array(aucs)
    best_f1  = int(np.argmax(cv_means))
    best_auc = int(np.argmax(aucs))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Layer Sweep — {model}  pca={pca_n}", fontsize=13)

    layers = list(range(n_layers))
    ax1.plot(layers, cv_means, "o-", color="#2ecc71", label="CV F1")
    ax1.fill_between(layers, cv_means - cv_stds, cv_means + cv_stds,
                     alpha=0.2, color="#2ecc71")
    ax1.axvline(best_f1, color="#2ecc71", linestyle="--",
                label=f"best layer={best_f1} ({cv_means[best_f1]:.3f})")
    ax1.set_ylabel("CV F1 (steer class)"); ax1.legend(); ax1.grid(True)

    ax2.plot(layers, aucs, "o-", color="#3498db", label="CV AUC")
    ax2.axvline(best_auc, color="#3498db", linestyle="--",
                label=f"best layer={best_auc} ({aucs[best_auc]:.3f})")
    ax2.set_ylabel("CV ROC-AUC"); ax2.set_xlabel("Layer")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    out_path = out_dir / f"layer_sweep_{model}_pca{pca_n}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Best layer by F1 : {best_f1} (F1={cv_means[best_f1]:.3f})")
    print(f"  Best layer by AUC: {best_auc} (AUC={aucs[best_auc]:.3f})")
    print(f"  Saved: {out_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--layer",   type=int, default=MIDDLE_LAYER)
    parser.add_argument("--pca",     type=int, default=0,
                        help="PCA components (0=disabled)")
    parser.add_argument("--ratio",   type=float, default=NO_CHANGE_RATIO,
                        help="class-0 downsample ratio (default: 1.0)")
    parser.add_argument("--C",       type=float, default=1.0,
                        help="LR regularization strength")
    parser.add_argument("--train",   default=None,
                        help="Path to train .npz (question-level split, from prepare_samples.py)")
    parser.add_argument("--test",    default=None,
                        help="Path to test .npz (question-level split, from prepare_samples.py)")
    parser.add_argument("--samples", default=None,
                        help="Legacy: single .npz with sample-level split fallback")
    parser.add_argument("--visualize",      action="store_true")
    parser.add_argument("--learning_curve", action="store_true")
    parser.add_argument("--layer_sweep",    action="store_true",
                        help="Sweep all layers and plot CV F1 / AUC")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for steer class (default: 0.5)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  ConfSteer Binary Classifier")
    print(f"  Model : {args.model}  Layer : {args.layer}")
    print(f"  PCA   : {args.pca if args.pca > 0 else 'disabled'}")
    print(f"  Ratio : {args.ratio}  C : {args.C}")
    print(f"{'='*50}\n")

    # ── Resolve paths ──
    # Mode A: question-level split (train + test npz)
    # Mode B: legacy single npz (sample-level split)
    train_path = Path(args.train) if args.train else None
    test_path  = Path(args.test)  if args.test  else None
    use_split  = train_path and train_path.exists()

    if not use_split:
        # fallback to legacy single npz
        if args.samples:
            samples_path = Path(args.samples)
        else:
            samples_path = SAMPLE_DIR / args.model / "samples_binary_all.npz"
            if samples_path.exists():
                print(f"  [auto] Using legacy npz: {samples_path}")
            else:
                samples_path = None

    # ── Layer Sweep ──
    if args.layer_sweep:
        print("\n[sweep] Sweeping all layers...")
        if use_split:
            X_all, y_all = load_samples(train_path)
            X_all = X_all.astype(np.float32)
            plot_layer_sweep_from_array(X_all, y_all, args.model, args.pca, args.ratio, args.C,
                                        BASE_DIR / "plots", already_balanced=True)
        elif samples_path and samples_path.exists():
            X_all, y_all = load_samples(samples_path)
            X_all = X_all.astype(np.float32)
            plot_layer_sweep_from_array(X_all, y_all, args.model, args.pca, args.ratio, args.C,
                                        BASE_DIR / "plots", already_balanced=True)
        else:
            print("[1] Loading labels (legacy H5 mode)...")
            label_files = load_labels(args.model)
            X_all, y_all = extract_all_layers(args.model, label_files)
            plot_layer_sweep_from_array(X_all, y_all, args.model, args.pca, args.ratio, args.C,
                                        BASE_DIR / "plots")
        return

    # ── Single layer: load features ──
    if use_split:
        print("[1] Loading train/test samples (question-level split)...")
        X_tr_all, y_train = load_samples(train_path)
        X_te_all, y_test  = load_samples(test_path)
        layer = min(args.layer, X_tr_all.shape[1] - 1)
        X_train = X_tr_all[:, layer, :].astype(np.float32)
        X_test  = X_te_all[:, layer, :].astype(np.float32)
        print(f"  Using layer {layer} of {X_tr_all.shape[1]}")
        print(f"  Train — steer(1): {(y_train==1).sum()}, no_steer(0): {(y_train==0).sum()}")
        print(f"  Test  — steer(1): {(y_test==1).sum()},  no_steer(0): {(y_test==0).sum()}")
    elif samples_path and samples_path.exists():
        print("[1] Loading pre-extracted samples (legacy single npz)...")
        X_all, y_all = load_samples(samples_path)
        layer = min(args.layer, X_all.shape[1] - 1)
        X = X_all[:, layer, :].astype(np.float32)
        print(f"  Using layer {layer} of {X_all.shape[1]}")
        print(f"  Raw — steer(+1): {(y_all==1).sum()}, no_steer(0): {(y_all==0).sum()}")
        print(f"\n[3] Skipping balance (already downsampled in npz)  Total: {len(y_all)}")
        # fall through to legacy path below
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
        )
    else:
        print("[1] Loading labels (legacy H5 mode)...")
        label_files = load_labels(args.model)
        print(f"  Found {len(label_files)} label files")
        print("\n[2] Extracting features...")
        X, y = extract_features(args.model, label_files, args.layer)
        print(f"  Raw — steer(+1): {(y==1).sum()}, no_steer(0): {(y==0).sum()}")
        print("\n[3] Balancing...")
        X, y = balanced_sample(X, y, ratio=args.ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

    # ── Scale + PCA on train, apply to test ──
    print("\n[4] Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if args.pca > 0:
        pca = PCA(n_components=min(args.pca, X_train_s.shape[1]), random_state=RANDOM_SEED)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s  = pca.transform(X_test_s)
        var = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA({args.pca}): explained variance = {var:.1f}%")

    clf = LogisticRegression(
        max_iter=1000, C=args.C,
        class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    f1_scores  = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="f1")
    auc_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="roc_auc")
    print(f"  5-fold CV F1 : {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    print(f"  5-fold CV AUC: {auc_scores.mean():.3f}")

    clf.fit(X_train_s, y_train)

    if args.visualize:
        print("\n[viz] Generating plots...")
        visualize(X_train_s, y_train, args.model, args.layer, args.pca, BASE_DIR / "plots")

    print("\n[5] Evaluation on test set:")
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(int)
    if args.threshold != 0.5:
        print(f"  Using threshold: {args.threshold}")
    print(classification_report(y_test, y_pred,
                                 target_names=["no_steer(0)", "steer(+1)"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("  labels: 0, +1")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    if args.learning_curve:
        print("\n[6] Plotting learning curve...")
        plot_learning_curve(clf, X_train_s, y_train, args.model, args.layer, args.pca,
                            BASE_DIR / "plots")


if __name__ == "__main__":
    main()

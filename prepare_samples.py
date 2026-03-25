"""
prepare_samples.py — Pre-extract hidden states for all label entries.

Reads labels/*.json + HiddenStates H5 files, then saves:
  samples/{model}/binary_{layer_start}_{layer_end}.npz   (or all layers)
    X : (N, n_layers, hidden_dim)  float32
    y_binary : (N,)  int8   — 1 if label_pos4==1 else 0
    y_three  : (N,)  int8   — 0=no_change, 1=pos4_works, 2=neg4_works (label_pos4: 0→0, 1→1, -1→2)

Usage:
  python prepare_samples.py --model llama3
  python prepare_samples.py --model qwen3
  python prepare_samples.py --model llama3 --roles neutral confident expert
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# ==================== Paths ====================
BASE_DIR   = Path(__file__).parent
LABEL_DIR  = BASE_DIR / "labels"
HIDDEN_DIR = BASE_DIR / "HiddenStates"
SAMPLE_DIR = BASE_DIR / "samples"


# ==================== H5 stem helpers ====================

def label_stem_to_h5_stem(orig_stem: str) -> str:
    """Remove known suffixes to get the bare H5 stem."""
    for suffix in ["_answers", "_mc1", "_mc2"]:
        if orig_stem.endswith(suffix):
            return orig_stem[: -len(suffix)]
    return orig_stem


def get_h5_path(model: str, task: str, orig_stem: str, role: str) -> Path:
    """
    Map (model, task, orig_stem, role) → H5 path.

    Filename pattern:
      generic roles : {role}_{h5_stem}.h5
      e.g. neutral_GPQA_(gpqa_diamond)_8B.h5
           confident_Biology_8B.h5

    TruthfulQA task dir is 'truthfulqa' regardless of tqa_mc1/mc2.
    """
    h5_stem = label_stem_to_h5_stem(orig_stem)
    h5_task = "truthfulqa" if task.startswith("tqa") else task
    return HIDDEN_DIR / model / h5_task / f"{role}_{h5_stem}.h5"


# ==================== Label loading ====================

def load_labels(model: str, roles_filter=None):
    """
    Returns list of (task, orig_stem, entries) tuples.
    entries is the list of dicts from labels_*.json["data"].
    If roles_filter is given, only keep entries whose role is in the set.
    """
    result = []
    pattern = str(LABEL_DIR / model / "**" / "labels_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        p = Path(path)
        task = p.parent.name
        orig_stem = p.stem[len("labels_"):]
        with open(p) as f:
            entries = json.load(f)["data"]
        if roles_filter:
            entries = [e for e in entries if e["role"] in roles_filter]
        if entries:
            result.append((task, orig_stem, entries))
    return result


# ==================== Core extraction ====================

def extract(model: str, label_files, roles_filter=None):
    """
    For each label entry, load the corresponding row from its role H5.
    Returns:
      X        : (N, n_layers, hidden_dim) float32
      y_binary : (N,) int8
      y_three  : (N,) int8  — 0=no_change, 1=pos4_works, 2=neg4_works
      meta     : list of dicts with task/orig_stem/role/index per row
    """
    import h5py

    X_list, y_bin_list, y_thr_list, meta_list = [], [], [], []
    skipped_files = set()

    for task, orig_stem, entries in label_files:
        # Group entries by role to open each H5 only once
        by_role = {}
        for e in entries:
            by_role.setdefault(e["role"], []).append(e)

        for role, role_entries in by_role.items():
            h5_path = get_h5_path(model, task, orig_stem, role)
            if not h5_path.exists():
                if h5_path not in skipped_files:
                    print(f"  [skip] H5 not found: {h5_path}")
                    skipped_files.add(h5_path)
                continue

            with h5py.File(h5_path, "r") as hf:
                hs = hf["hidden_states"]  # (N_q, n_layers, hidden_dim)
                for e in role_entries:
                    idx = e["index"]
                    if idx >= hs.shape[0]:
                        print(f"  [warn] index {idx} out of range for {h5_path.name} (len={hs.shape[0]})")
                        continue
                    X_list.append(hs[idx, :, :])          # (n_layers, hidden_dim)
                    y_bin_list.append(1 if e["label_pos4"] == 1 else 0)
                    lp = e["label_pos4"]
                    y_thr_list.append(2 if lp == -1 else lp)  # -1→2, 0→0, 1→1
                    meta_list.append({
                        "task":      task,
                        "orig_stem": orig_stem,
                        "role":      role,
                        "index":     idx,
                    })

    if not X_list:
        raise RuntimeError("No samples extracted — check H5 paths and label files.")

    X        = np.array(X_list, dtype=np.float16)
    y_binary = np.array(y_bin_list, dtype=np.int8)
    y_three  = np.array(y_thr_list, dtype=np.int8)
    print(f"  Extracted {len(X)} samples, shape: {X.shape}")
    print(f"  y_binary — steer(1): {(y_binary==1).sum()}, no_steer(0): {(y_binary==0).sum()}")
    print(f"  y_three  — 1(pos4): {(y_three==1).sum()}, 0(no_change): {(y_three==0).sum()}, 2(neg4): {(y_three==2).sum()}")
    return X, y_binary, y_three, meta_list


# ==================== Downsample class 0 ====================

def downsample_class0(X, y_binary, y_three, meta_list, ratio: float, seed: int = 42):
    """
    Keep all minority samples (class1=pos4, class2=neg4).
    Randomly keep ratio × max(n_class1, n_class2) samples from class0.
    Uses y_three to identify classes; y_binary is kept in sync.
    """
    rng = np.random.default_rng(seed)

    idx_c1   = np.where(y_three == 1)[0]
    idx_c2   = np.where(y_three == 2)[0]
    idx_c0   = np.where(y_three == 0)[0]

    n_minority = max(len(idx_c1), len(idx_c2))
    n_keep_c0  = int(ratio * n_minority)
    n_keep_c0  = min(n_keep_c0, len(idx_c0))

    idx_c0_kept = rng.choice(idx_c0, size=n_keep_c0, replace=False)
    keep = np.concatenate([idx_c1, idx_c2, idx_c0_kept])
    keep.sort()

    meta_kept = [meta_list[i] for i in keep]
    print(f"  Downsampled class0: {len(idx_c0)} → {n_keep_c0}  (ratio={ratio})")
    print(f"  Final — 1(pos4): {len(idx_c1)}, 2(neg4): {len(idx_c2)}, 0(no_change): {n_keep_c0}  total: {len(keep)}")
    return X[keep], y_binary[keep], y_three[keep], meta_kept


# ==================== Save / Load ====================

def save_samples(out_path: Path, X, y_binary, y_three, meta_list, roles):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # meta as JSON string array (npz can't store dicts natively)
    meta_json = np.array([json.dumps(m) for m in meta_list])
    roles_arr = np.array(roles if roles else ["all"])
    np.savez_compressed(
        out_path,
        X=X,
        y_binary=y_binary,
        y_three=y_three,
        meta=meta_json,
        roles=roles_arr,
    )
    print(f"  Saved → {out_path}  ({out_path.stat().st_size // 1024 // 1024} MB)")


def load_samples(path: Path):
    data = np.load(path, allow_pickle=False)
    X        = data["X"]
    y_binary = data["y_binary"]
    y_three  = data["y_three"]
    meta     = [json.loads(s) for s in data["meta"]]
    roles    = list(data["roles"])
    return X, y_binary, y_three, meta, roles


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Pre-extract hidden states for ConfSteer classifiers")
    parser.add_argument("--model",  default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--roles",  nargs="*", default=None,
                        help="Roles to include (default: all). E.g. --roles neutral confident expert")
    parser.add_argument("--out",    default=None,
                        help="Output .npz path (default: samples/{model}/samples_{roles}.npz)")
    parser.add_argument("--ratio",  type=float, default=1.0,
                        help="class-0 downsample ratio relative to max(n_class1, n_class2) (default: 1.0)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    roles_filter = set(args.roles) if args.roles else None
    roles_tag    = "_".join(sorted(args.roles)) if args.roles else "all"

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = SAMPLE_DIR / args.model / f"samples_{roles_tag}.npz"

    print(f"\n{'='*50}")
    print(f"  prepare_samples")
    print(f"  Model : {args.model}")
    print(f"  Roles : {roles_tag}")
    print(f"  Ratio : {args.ratio}  Seed: {args.seed}")
    print(f"  Out   : {out_path}")
    print(f"{'='*50}\n")

    print("[1] Loading labels...")
    label_files = load_labels(args.model, roles_filter=roles_filter)
    total_entries = sum(len(e) for _, _, e in label_files)
    print(f"  {len(label_files)} label files, {total_entries} entries")

    print("\n[2] Extracting hidden states...")
    X, y_binary, y_three, meta = extract(args.model, label_files, roles_filter)

    print("\n[3] Downsampling class 0...")
    X, y_binary, y_three, meta = downsample_class0(X, y_binary, y_three, meta, args.ratio, args.seed)

    print("\n[4] Saving...")
    save_samples(out_path, X, y_binary, y_three, meta,
                 roles=args.roles if args.roles else [])

    print("\nDone.")


if __name__ == "__main__":
    main()

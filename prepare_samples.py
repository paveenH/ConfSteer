"""
prepare_samples.py — Pre-extract hidden states for all label entries.

Reads labels/*.json + HiddenStates H5 files, then saves TWO npz files:

  samples/{model}/samples_binary_{roles}.npz
    X        : (N, n_layers, hidden_dim)  float16
    y        : (N,)  int8   — 1=steer(pos4 works), 0=no_steer(all others)
    meta     : (N,)  JSON strings
    roles    : role list
    class0 = y==0 downsampled from (no_change + neg4_works), ratio × n_class1

  samples/{model}/samples_three_{roles}.npz
    X        : (N, n_layers, hidden_dim)  float16
    y        : (N,)  int8   — 0=no_change, 1=pos4_works, 2=neg4_works
    meta     : (N,)  JSON strings
    roles    : role list
    class0 = y==0 (no_change only) downsampled, ratio × max(n_class1, n_class2)

Usage:
  python prepare_samples.py --model llama3
  python prepare_samples.py --model qwen3 --ratio 1.0
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
      X       : (N, n_layers, hidden_dim) float16
      y_three : (N,) int8  — 0=no_change, 1=pos4_works, 2=neg4_works
      meta    : list of dicts with task/orig_stem/role/index per row
    """
    import h5py

    X_list, y_thr_list, meta_list = [], [], []
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

    X       = np.array(X_list, dtype=np.float16)
    y_three = np.array(y_thr_list, dtype=np.int8)
    print(f"  Extracted {len(X)} samples, shape: {X.shape}")
    print(f"  y_three  — 1(pos4): {(y_three==1).sum()}, 0(no_change): {(y_three==0).sum()}, 2(neg4): {(y_three==2).sum()}")
    return X, y_three, meta_list


# ==================== Downsample ====================

def downsample_binary(X, y_three, meta_list, ratio: float, seed: int):
    """
    Binary: class1 = pos4_works (y_three==1), class0 = everything else.
    Downsample class0 to ratio × n_class1.
    """
    rng = np.random.default_rng(seed)
    idx_c1 = np.where(y_three == 1)[0]
    idx_c0 = np.where(y_three != 1)[0]   # no_change + neg4_works

    n_keep = min(int(ratio * len(idx_c1)), len(idx_c0))
    idx_c0_kept = rng.choice(idx_c0, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([idx_c1, idx_c0_kept]))

    y = (y_three[keep] == 1).astype(np.int8)
    print(f"  [binary] class1(pos4): {len(idx_c1)}, class0(other): {len(idx_c0)} → {n_keep}  total: {len(keep)}")
    return X[keep], y, [meta_list[i] for i in keep]


def downsample_three(X, y_three, meta_list, ratio: float, seed: int):
    """
    Three-class: 0=no_change, 1=pos4_works, 2=neg4_works.
    Downsample class0 (no_change only) to ratio × max(n_class1, n_class2).
    """
    rng = np.random.default_rng(seed)
    idx_c1 = np.where(y_three == 1)[0]
    idx_c2 = np.where(y_three == 2)[0]
    idx_c0 = np.where(y_three == 0)[0]

    n_keep = min(int(ratio * max(len(idx_c1), len(idx_c2))), len(idx_c0))
    idx_c0_kept = rng.choice(idx_c0, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([idx_c1, idx_c2, idx_c0_kept]))

    print(f"  [three]  1(pos4): {len(idx_c1)}, 2(neg4): {len(idx_c2)}, 0(no_change): {len(idx_c0)} → {n_keep}  total: {len(keep)}")
    return X[keep], y_three[keep], [meta_list[i] for i in keep]


# ==================== Save / Load ====================

def save_npz(out_path: Path, X, y, meta_list, roles):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        meta=np.array([json.dumps(m) for m in meta_list]),
        roles=np.array(roles if roles else ["all"]),
    )
    print(f"  Saved → {out_path}  ({out_path.stat().st_size // 1024 // 1024} MB)")


def load_samples(path: Path):
    """Load a binary or three-class npz. Returns X, y, meta, roles."""
    data = np.load(path, allow_pickle=False)
    X     = data["X"]
    y     = data["y"]
    meta  = [json.loads(s) for s in data["meta"]]
    roles = list(data["roles"])
    return X, y, meta, roles


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

    out_dir = SAMPLE_DIR / args.model

    print(f"\n{'='*50}")
    print(f"  prepare_samples")
    print(f"  Model : {args.model}")
    print(f"  Roles : {roles_tag}")
    print(f"  Ratio : {args.ratio}  Seed: {args.seed}")
    print(f"  Out   : {out_dir}/samples_binary_{roles_tag}.npz")
    print(f"          {out_dir}/samples_three_{roles_tag}.npz")
    print(f"{'='*50}\n")

    print("[1] Loading labels...")
    label_files = load_labels(args.model, roles_filter=roles_filter)
    total_entries = sum(len(e) for _, _, e in label_files)
    print(f"  {len(label_files)} label files, {total_entries} entries")

    print("\n[2] Extracting hidden states...")
    X, y_three, meta = extract(args.model, label_files, roles_filter)

    roles_list = args.roles if args.roles else []

    print("\n[3a] Downsampling → binary...")
    X_bin, y_bin, meta_bin = downsample_binary(X, y_three, meta, args.ratio, args.seed)

    print("\n[3b] Downsampling → three-class...")
    X_thr, y_thr, meta_thr = downsample_three(X, y_three, meta, args.ratio, args.seed)

    print("\n[4] Saving...")
    save_npz(out_dir / f"samples_binary_{roles_tag}.npz", X_bin, y_bin, meta_bin, roles_list)
    save_npz(out_dir / f"samples_three_{roles_tag}.npz",  X_thr, y_thr, meta_thr, roles_list)

    print("\nDone.")


if __name__ == "__main__":
    main()

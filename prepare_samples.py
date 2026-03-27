"""
prepare_samples.py — Pre-extract hidden states for all label entries.

Reads labels/*.json + HiddenStates H5 files, then saves train/test npz pairs:

  samples/{model}/samples_binary_{roles}_train.npz
  samples/{model}/samples_binary_{roles}_test.npz
    X     : (N, n_layers, hidden_dim)  float16
    y     : (N,)  int8   — 1=steer(pos4 works), 0=no_steer(all others)
    meta  : (N,)  JSON strings  {"task", "orig_stem", "role", "index"}
    roles : role list

  samples/{model}/samples_three_{roles}_train.npz
  samples/{model}/samples_three_{roles}_test.npz
    X     : (N, n_layers, hidden_dim)  float16
    y     : (N,)  int8   — 0=no_change, 1=pos4_works, 2=neg4_works
    meta  : (N,)  JSON strings
    roles : role list

Split is question-level: (task, orig_stem, index) uniquely identifies a question.
All roles of the same question go to the same split (train or test).
  - train: class0 downsampled to ratio × n_class1
  - test : original distribution (no downsample)

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
    h5_stem = label_stem_to_h5_stem(orig_stem)
    h5_task = "truthfulqa" if task.startswith("tqa") else task
    return HIDDEN_DIR / model / h5_task / f"{role}_{h5_stem}.h5"


# ==================== Label loading ====================

def load_labels(model: str, roles_filter=None):
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
      X        : (N, n_layers, hidden_dim) float16
      y_three  : (N,) int8  — 0=no_change, 1=pos4_works, 2=neg4_works
      y_orig   : (N,) int8  — 1=orig_correct, 0=orig_wrong
      meta     : list of dicts with task/orig_stem/role/index per row
    """
    import h5py

    X_list, y_thr_list, y_orig_list, meta_list = [], [], [], []
    skipped_files = set()

    for task, orig_stem, entries in label_files:
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
                    X_list.append(hs[idx, :, :])
                    lp = e["label_pos4"]
                    y_thr_list.append(2 if lp == -1 else lp)  # -1→2, 0→0, 1→1
                    y_orig_list.append(1 if e["orig_correct"] else 0)
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
    y_orig  = np.array(y_orig_list, dtype=np.int8)
    print(f"  Extracted {len(X)} samples, shape: {X.shape}")
    print(f"  y_three — 1(pos4): {(y_three==1).sum()}, 0(no_change): {(y_three==0).sum()}, 2(neg4): {(y_three==2).sum()}")
    print(f"  y_orig  — 1(correct): {(y_orig==1).sum()}, 0(wrong): {(y_orig==0).sum()}")
    return X, y_three, y_orig, meta_list


# ==================== Question-level split ====================

def question_level_split(y_three, meta_list, test_size: float, seed: int):
    """
    Split by unique question ID = (task, orig_stem, index).
    Stratify by majority label of each question (across roles).

    Returns train_mask, test_mask  (bool arrays of length N).
    """
    from collections import defaultdict

    # Build question → sample indices mapping
    qid_to_indices = defaultdict(list)
    for i, m in enumerate(meta_list):
        qid = (m["task"], m["orig_stem"], m["index"])
        qid_to_indices[qid].append(i)

    # Assign a label to each question: majority label_pos4 across its roles
    # y_three==1 means pos4_works; use that as the stratify signal
    qids = list(qid_to_indices.keys())
    q_labels = []
    for qid in qids:
        idxs = qid_to_indices[qid]
        has_pos4 = any(y_three[i] == 1 for i in idxs)
        q_labels.append(1 if has_pos4 else 0)
    q_labels = np.array(q_labels)

    # Stratified split of questions
    from sklearn.model_selection import train_test_split
    q_train, q_test = train_test_split(
        np.arange(len(qids)),
        test_size=test_size,
        random_state=seed,
        stratify=q_labels,
    )

    train_q_set = {qids[i] for i in q_train}
    test_q_set  = {qids[i] for i in q_test}

    train_mask = np.zeros(len(meta_list), dtype=bool)
    test_mask  = np.zeros(len(meta_list), dtype=bool)
    for i, m in enumerate(meta_list):
        qid = (m["task"], m["orig_stem"], m["index"])
        if qid in train_q_set:
            train_mask[i] = True
        else:
            test_mask[i] = True

    n_train_q = len(train_q_set)
    n_test_q  = len(test_q_set)
    print(f"  Question-level split: {n_train_q} train questions, {n_test_q} test questions")
    print(f"  Sample split: {train_mask.sum()} train, {test_mask.sum()} test")
    return train_mask, test_mask


# ==================== Downsample (train only) ====================

def downsample_binary(X, y_three, meta_list, ratio: float, seed: int):
    """
    Binary: class1 = pos4_works (y_three==1), class0 = everything else.
    Downsample class0 to ratio × n_class1.
    """
    rng = np.random.default_rng(seed)
    idx_c1 = np.where(y_three == 1)[0]
    idx_c0 = np.where(y_three != 1)[0]

    n_keep = min(int(ratio * len(idx_c1)), len(idx_c0))
    idx_c0_kept = rng.choice(idx_c0, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([idx_c1, idx_c0_kept]))

    y = (y_three[keep] == 1).astype(np.int8)
    print(f"  [binary] class1(pos4): {len(idx_c1)}, class0(other): {len(idx_c0)} → {n_keep}  total: {len(keep)}")
    return X[keep], y, [meta_list[i] for i in keep]


def downsample_orig(X, y_orig, meta_list, ratio: float, seed: int, max_per_class: int = None):
    """
    orig_correct binary: class1 = correct (y_orig==1), class0 = wrong.
    Downsample the majority class to ratio × n_minority.
    If max_per_class is set, further cap each class to that many samples.
    """
    rng = np.random.default_rng(seed)
    idx_c1 = np.where(y_orig == 1)[0]
    idx_c0 = np.where(y_orig == 0)[0]

    # downsample the majority class
    if len(idx_c1) >= len(idx_c0):
        n_keep = min(int(ratio * len(idx_c0)), len(idx_c1))
        idx_c1_kept = rng.choice(idx_c1, size=n_keep, replace=False)
        idx_c0_kept = idx_c0
    else:
        n_keep = min(int(ratio * len(idx_c1)), len(idx_c0))
        idx_c0_kept = rng.choice(idx_c0, size=n_keep, replace=False)
        idx_c1_kept = idx_c1

    # cap each class if max_per_class is specified
    if max_per_class is not None:
        if len(idx_c1_kept) > max_per_class:
            idx_c1_kept = rng.choice(idx_c1_kept, size=max_per_class, replace=False)
        if len(idx_c0_kept) > max_per_class:
            idx_c0_kept = rng.choice(idx_c0_kept, size=max_per_class, replace=False)

    keep = np.sort(np.concatenate([idx_c1_kept, idx_c0_kept]))
    print(f"  [orig] correct(1): {len(idx_c1)}, wrong(0): {len(idx_c0)} → kept: {len(idx_c1_kept)}+{len(idx_c0_kept)}={len(keep)}")
    return X[keep], y_orig[keep], [meta_list[i] for i in keep]


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


# ==================== Save ====================

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


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Pre-extract hidden states for ConfSteer classifiers")
    parser.add_argument("--model",  default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--roles",  nargs="*", default=None,
                        help="Roles to include (default: all). E.g. --roles neutral confident expert")
    parser.add_argument("--ratio",  type=float, default=1.0,
                        help="class-0 downsample ratio for train set (default: 1.0)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of questions for test set (default: 0.2)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--task",   nargs="*", default=None,
                        choices=["binary", "three", "orig"],
                        help="Which outputs to save (default: all). E.g. --task orig")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap each class to this many samples (orig task only). E.g. --max_per_class 1000")
    args = parser.parse_args()

    roles_filter = set(args.roles) if args.roles else None
    roles_tag    = "_".join(sorted(args.roles)) if args.roles else "all"
    out_dir      = SAMPLE_DIR / args.model
    tasks        = set(args.task) if args.task else {"binary", "three", "orig"}

    print(f"\n{'='*55}")
    print(f"  prepare_samples")
    print(f"  Model     : {args.model}")
    print(f"  Roles     : {roles_tag}")
    print(f"  Tasks     : {', '.join(sorted(tasks))}")
    print(f"  Ratio     : {args.ratio}  Test size: {args.test_size}  Seed: {args.seed}")
    print(f"  Out       : {out_dir}/samples_{{{'|'.join(sorted(tasks))}}}_{roles_tag}_{{train,test}}.npz")
    print(f"{'='*55}\n")

    print("[1] Loading labels...")
    label_files = load_labels(args.model, roles_filter=roles_filter)
    total_entries = sum(len(e) for _, _, e in label_files)
    print(f"  {len(label_files)} label files, {total_entries} entries")

    print("\n[2] Extracting hidden states...")
    X, y_three, y_orig, meta = extract(args.model, label_files, roles_filter)

    print("\n[3] Question-level train/test split...")
    train_mask, test_mask = question_level_split(y_three, meta, args.test_size, args.seed)

    X_tr, y_tr, y_orig_tr, meta_tr = X[train_mask], y_three[train_mask], y_orig[train_mask], [meta[i] for i in np.where(train_mask)[0]]
    X_te, y_te, y_orig_te, meta_te = X[test_mask],  y_three[test_mask],  y_orig[test_mask],  [meta[i] for i in np.where(test_mask)[0]]

    roles_list = args.roles if args.roles else []

    print("\n[5] Saving...")
    if "binary" in tasks:
        print("\n[4a] Downsampling train → binary...")
        X_bin_tr, y_bin_tr, meta_bin_tr = downsample_binary(X_tr, y_tr, meta_tr, args.ratio, args.seed)
        print("\n[4b] Preparing test → binary (no downsample)...")
        y_bin_te = (y_te == 1).astype(np.int8)
        print(f"  [binary test] class1(pos4): {(y_bin_te==1).sum()}, class0(other): {(y_bin_te==0).sum()}  total: {len(y_bin_te)}")
        save_npz(out_dir / f"samples_binary_{roles_tag}_train.npz", X_bin_tr, y_bin_tr,  meta_bin_tr, roles_list)
        save_npz(out_dir / f"samples_binary_{roles_tag}_test.npz",  X_te,     y_bin_te,  meta_te,     roles_list)

    if "three" in tasks:
        print("\n[4c] Downsampling train → three-class...")
        X_thr_tr, y_thr_tr, meta_thr_tr = downsample_three(X_tr, y_tr, meta_tr, args.ratio, args.seed)
        print("\n[4d] Preparing test → three-class (no downsample)...")
        print(f"  [three test]  1(pos4): {(y_te==1).sum()}, 2(neg4): {(y_te==2).sum()}, 0(no_change): {(y_te==0).sum()}  total: {len(y_te)}")
        save_npz(out_dir / f"samples_three_{roles_tag}_train.npz", X_thr_tr, y_thr_tr, meta_thr_tr, roles_list)
        save_npz(out_dir / f"samples_three_{roles_tag}_test.npz",  X_te,     y_te,     meta_te,     roles_list)

    if "orig" in tasks:
        print("\n[4e] Downsampling train → orig_correct...")
        X_orig_tr, y_orig_tr_ds, meta_orig_tr = downsample_orig(X_tr, y_orig_tr, meta_tr, args.ratio, args.seed, args.max_per_class)
        print("\n[4f] Preparing test → orig_correct (no downsample)...")
        print(f"  [orig test]  correct(1): {(y_orig_te==1).sum()}, wrong(0): {(y_orig_te==0).sum()}  total: {len(y_orig_te)}")
        save_npz(out_dir / f"samples_orig_{roles_tag}_train.npz",  X_orig_tr, y_orig_tr_ds, meta_orig_tr, roles_list)
        save_npz(out_dir / f"samples_orig_{roles_tag}_test.npz",   X_te,      y_orig_te,    meta_te,      roles_list)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
prepare_samples_mmlu.py — Build orig_correct samples from MMLU answer JSONs + H5 files.
=========================================================================================
Reads:
  <answer_dir>/{model}/mmlu/{task}_{size}_answers.json   — orig answers per role
  HiddenStates/{model}/mmlu/{role_key}_{task}_{size}.h5  — hidden states

Writes (will NOT overwrite existing prepare_samples.py outputs):
  samples/{model}/samples_orig_mmlu_{tag}_train.npz
  samples/{model}/samples_orig_mmlu_{tag}_test.npz
    X     : (N, n_layers, hidden_dim)  float16
    y     : (N,)  int8   — 1=orig_correct, 0=orig_wrong
    meta  : (N,)  JSON strings  {"task", "role", "index"}
    roles : role list

Split is question-level: all roles of the same question go to the same split.

Usage:
  python prepare_samples_mmlu.py --model llama3 --size 8B
  python prepare_samples_mmlu.py --model qwen3  --size 8B --tag 5k --max_per_class 5000
  python prepare_samples_mmlu.py --model llama3 --size 8B --roles neutral confident
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR    = Path(__file__).parent
HIDDEN_DIR  = BASE_DIR / "HiddenStates"
SAMPLE_DIR  = BASE_DIR / "samples"

LABELS_4 = ["A", "B", "C", "D"]

# Role key mapping: role name as stored in JSON answer keys
# e.g. "neutral" → "answer_neutral", "{task} expert" → "answer_{task}_expert"
def role_to_key(role: str, task: str) -> str:
    """Convert role name to the answer key used in the JSON."""
    task_slug = task.replace(" ", "_")
    r = role.replace("{task}", task_slug)
    return "answer_" + r.replace(" ", "_")


def role_to_h5_prefix(role: str, task: str) -> str:
    """Convert role name to the H5 filename prefix."""
    task_slug = task.replace(" ", "_")
    r = role.replace("{task}", task_slug)
    return r.replace(" ", "_")


# ==================== Extract ====================

def extract(model: str, size: str, answer_dir: Path, roles_filter: list):
    """
    Read all answer JSONs + H5 files and build (X, y_orig, meta).
    Returns:
      X       : list of float16 arrays (n_layers, hidden_dim)
      y_orig  : list of int8  — 1=correct, 0=wrong
      meta    : list of dicts {"task", "role", "index"}
    """
    X_list, y_list, meta_list = [], [], []
    skipped_h5 = set()

    ans_files = sorted((answer_dir / model / "mmlu").glob(f"*_{size}_answers.json"))
    if not ans_files:
        raise FileNotFoundError(f"No answer files found in {answer_dir / model / 'mmlu'}")

    print(f"  Found {len(ans_files)} answer files")

    for ans_path in ans_files:
        with open(ans_path, encoding="utf-8") as f:
            d = json.load(f)
        samples = d["data"]
        if not samples:
            continue

        # Infer task name from first sample
        task = samples[0]["task"]
        task_slug = task.replace(" ", "_")

        for role in roles_filter:
            ans_key = role_to_key(role, task)
            h5_prefix = role_to_h5_prefix(role, task)
            h5_path = HIDDEN_DIR / model / "mmlu" / f"{h5_prefix}_{task_slug}_{size}.h5"

            if not h5_path.exists():
                if h5_path not in skipped_h5:
                    print(f"  [skip] H5 not found: {h5_path}")
                    skipped_h5.add(h5_path)
                continue

            with h5py.File(h5_path, "r") as hf:
                hs = hf["hidden_states"]  # (N, n_layers, hidden_dim)
                n_h5 = hs.shape[0]

                for idx, sample in enumerate(samples):
                    if idx >= n_h5:
                        print(f"  [warn] index {idx} out of range for {h5_path.name}")
                        break

                    pred = sample.get(ans_key)
                    if pred is None:
                        continue

                    true_label = LABELS_4[int(sample["label"])]
                    orig_correct = int(pred == true_label)

                    X_list.append(hs[idx, :, :])  # read row lazily
                    y_list.append(orig_correct)
                    meta_list.append({"task": task_slug, "role": role, "index": idx})

    if not X_list:
        raise RuntimeError("No samples extracted — check paths and role names.")

    X      = np.array(X_list, dtype=np.float16)
    y_orig = np.array(y_list,  dtype=np.int8)
    print(f"  Extracted {len(X)} samples, shape: {X.shape}")
    print(f"  y_orig — correct(1): {(y_orig==1).sum()}, wrong(0): {(y_orig==0).sum()}")
    return X, y_orig, meta_list


# ==================== Question-level split ====================

def question_level_split(y_orig, meta_list, test_size: float, seed: int):
    """Split by (task, index) so all roles of the same question stay together."""
    qid_to_indices = defaultdict(list)
    for i, m in enumerate(meta_list):
        qid = (m["task"], m["index"])
        qid_to_indices[qid].append(i)

    qids = list(qid_to_indices.keys())
    # Stratify by orig_correct majority across roles
    q_labels = []
    for qid in qids:
        idxs = qid_to_indices[qid]
        majority = int(np.mean([y_orig[i] for i in idxs]) >= 0.5)
        q_labels.append(majority)
    q_labels = np.array(q_labels)

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
        qid = (m["task"], m["index"])
        if qid in train_q_set:
            train_mask[i] = True
        else:
            test_mask[i] = True

    print(f"  Question-level split: {len(train_q_set)} train Qs, {len(test_q_set)} test Qs")
    print(f"  Sample split: {train_mask.sum()} train, {test_mask.sum()} test")
    return train_mask, test_mask


# ==================== Downsample ====================

def downsample_orig(X, y, meta_list, ratio: float, seed: int, max_per_class: int = None):
    """Downsample majority class to ratio × minority; optionally cap each class."""
    rng = np.random.default_rng(seed)
    idx_c1 = np.where(y == 1)[0]
    idx_c0 = np.where(y == 0)[0]

    if len(idx_c1) >= len(idx_c0):
        n_keep = min(int(ratio * len(idx_c0)), len(idx_c1))
        idx_c1 = rng.choice(idx_c1, size=n_keep, replace=False)
    else:
        n_keep = min(int(ratio * len(idx_c1)), len(idx_c0))
        idx_c0 = rng.choice(idx_c0, size=n_keep, replace=False)

    if max_per_class is not None:
        if len(idx_c1) > max_per_class:
            idx_c1 = rng.choice(idx_c1, size=max_per_class, replace=False)
        if len(idx_c0) > max_per_class:
            idx_c0 = rng.choice(idx_c0, size=max_per_class, replace=False)

    keep = np.sort(np.concatenate([idx_c1, idx_c0]))
    print(f"  [orig] correct(1): {(y==1).sum()}, wrong(0): {(y==0).sum()} → kept {len(idx_c1)}+{len(idx_c0)}={len(keep)}")
    return X[keep], y[keep], [meta_list[i] for i in keep]


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
    size_mb = out_path.stat().st_size // 1024 // 1024
    print(f"  Saved → {out_path}  ({size_mb} MB)")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Prepare MMLU orig_correct samples for ConfSteer classifier")
    parser.add_argument("--model",   required=True, choices=["llama3", "qwen3"])
    parser.add_argument("--size",    default="8B", help="Model size, e.g. 8B")
    parser.add_argument("--roles",   nargs="*", default=None,
                        help="Roles to include (default: all 7). Use {task} as placeholder. "
                             "E.g. --roles neutral confident '{task} expert'")
    parser.add_argument("--answer_dir", type=str, default=None,
                        help="Base dir containing {model}/mmlu/ answer JSONs. "
                             "Defaults to <ConfSteer>/answer/")
    parser.add_argument("--ratio",   type=float, default=1.0,
                        help="Majority-class downsample ratio for train set (default: 1.0)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap each class in train set. E.g. --max_per_class 5000")
    parser.add_argument("--max_test_per_class", type=int, default=None,
                        help="Cap each class in test set.")
    parser.add_argument("--tag",     type=str, default=None,
                        help="Extra suffix for output filename. E.g. --tag 5k → samples_orig_mmlu_all_5k_train.npz")
    args = parser.parse_args()

    # ── Default roles ──
    DEFAULT_ROLES = [
        "neutral",
        "{task} expert",
        "non {task} expert",
        "confident",
        "unconfident",
        "student",
        "person",
    ]
    roles = args.roles if args.roles else DEFAULT_ROLES
    roles_base = "_".join(r.replace(" ", "_").replace("{task}", "task") for r in roles)
    if len(roles) == len(DEFAULT_ROLES) and set(roles) == set(DEFAULT_ROLES):
        roles_base = "all"
    roles_tag = f"{roles_base}_{args.tag}" if args.tag else roles_base
    roles_tag_train = roles_tag
    roles_tag_test  = roles_tag

    answer_dir = Path(args.answer_dir) if args.answer_dir else BASE_DIR / "answer"
    out_dir    = SAMPLE_DIR / args.model

    print(f"\n{'='*60}")
    print(f"  prepare_samples_mmlu")
    print(f"  Model      : {args.model} ({args.size})")
    print(f"  Roles      : {roles}")
    print(f"  Answer dir : {answer_dir / args.model / 'mmlu'}")
    print(f"  Hidden dir : {HIDDEN_DIR / args.model / 'mmlu'}")
    print(f"  Output tag : mmlu_{roles_tag}")
    print(f"{'='*60}\n")

    print("[1] Extracting samples from answer JSONs + H5 files...")
    X, y_orig, meta = extract(args.model, args.size, answer_dir, roles)

    print("\n[2] Question-level train/test split...")
    train_mask, test_mask = question_level_split(y_orig, meta, args.test_size, args.seed)

    X_tr      = X[train_mask]
    y_tr      = y_orig[train_mask]
    meta_tr   = [meta[i] for i in np.where(train_mask)[0]]

    X_te      = X[test_mask]
    y_te      = y_orig[test_mask]
    meta_te   = [meta[i] for i in np.where(test_mask)[0]]

    print("\n[3] Downsampling train set...")
    X_tr_ds, y_tr_ds, meta_tr_ds = downsample_orig(X_tr, y_tr, meta_tr, args.ratio, args.seed, args.max_per_class)

    print("\n[4] Preparing test set (original distribution)...")
    print(f"  [orig test] correct(1): {(y_te==1).sum()}, wrong(0): {(y_te==0).sum()}  total: {len(y_te)}")
    X_te_ds, y_te_ds, meta_te_ds = X_te, y_te, meta_te

    print("\n[5] Saving...")
    train_path = out_dir / f"samples_orig_mmlu_{roles_tag_train}_train.npz"
    test_path  = out_dir / f"samples_orig_mmlu_{roles_tag_test}_test.npz"
    save_npz(train_path, X_tr_ds, y_tr_ds, meta_tr_ds, roles)
    save_npz(test_path,  X_te_ds, y_te_ds, meta_te_ds, roles)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
merge_npz.py — Merge multiple samples_orig_*.npz files into one.
=================================================================
Usage:
  python merge_npz.py \
      --inputs samples/llama3/samples_orig_mmlu_all_train.npz \
                samples/llama3/samples_orig_mmlupro_all_train.npz \
      --output  samples/llama3/samples_orig_mmlu_mmlupro_all_train.npz

  # Merge both train and test in one call:
  python merge_npz.py \
      --inputs samples/llama3/samples_orig_mmlu_all_train.npz \
                samples/llama3/samples_orig_mmlupro_all_train.npz \
      --output  samples/llama3/samples_orig_mmlu_mmlupro_all_train.npz \
      --inputs2 samples/llama3/samples_orig_mmlu_all_test.npz \
                samples/llama3/samples_orig_mmlupro_all_test.npz \
      --output2 samples/llama3/samples_orig_mmlu_mmlupro_all_test.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np


def merge(paths: list[Path], out_path: Path):
    X_list, y_list, meta_list, roles_set = [], [], [], []

    for p in paths:
        print(f"  Loading {p.name} ...")
        data = np.load(p, allow_pickle=True)
        X_list.append(data["X"])
        y_list.append(data["y"])
        meta_list.extend(data["meta"].tolist())
        for r in data["roles"].tolist():
            if r not in roles_set:
                roles_set.append(r)
        print(f"    X: {data['X'].shape}, y: {data['y'].shape}, "
              f"correct(1): {(data['y']==1).sum()}, wrong(0): {(data['y']==0).sum()}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"\n  Merged: {X.shape}, correct(1): {(y==1).sum()}, wrong(0): {(y==0).sum()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        meta=np.array(meta_list),
        roles=np.array(roles_set),
    )
    size_mb = out_path.stat().st_size // 1024 // 1024
    print(f"  Saved → {out_path}  ({size_mb} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",  nargs="+", required=True, help="Input npz files to merge (set 1)")
    parser.add_argument("--output",  required=True, help="Output npz path (set 1)")
    parser.add_argument("--inputs2", nargs="+", default=None, help="Input npz files to merge (set 2, optional)")
    parser.add_argument("--output2", default=None, help="Output npz path (set 2, optional)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  merge_npz — set 1")
    print(f"{'='*55}")
    merge([Path(p) for p in args.inputs], Path(args.output))

    if args.inputs2:
        print(f"\n{'='*55}")
        print(f"  merge_npz — set 2")
        print(f"{'='*55}")
        merge([Path(p) for p in args.inputs2], Path(args.output2))

    print("\nDone.")


if __name__ == "__main__":
    main()

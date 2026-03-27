"""
analyze_orig_vs_steering.py
===========================
Analyze the relationship between orig_correct and label_pos4 (steering effectiveness).

Questions answered:
  1. Among wrong samples (orig_correct=0), what fraction benefit from +4 steering?
  2. Among correct samples (orig_correct=1), what fraction are harmed by +4 steering?
  3. If we steer all wrong-predicted samples, what is the expected net accuracy change?

Usage:
  python analyze_orig_vs_steering.py --model llama3
  python analyze_orig_vs_steering.py --model qwen3
"""

import argparse
import glob
import json
from pathlib import Path

BASE_DIR  = Path(__file__).parent
LABEL_DIR = BASE_DIR / "labels"


def load_all_entries(model: str, roles_filter=None):
    entries_all = []
    pattern = str(LABEL_DIR / model / "**" / "labels_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            data = json.load(f)["data"]
        if roles_filter:
            data = [e for e in data if e["role"] in roles_filter]
        entries_all.extend(data)
    return entries_all


def analyze(entries):
    total       = len(entries)
    orig_wrong  = [e for e in entries if not e["orig_correct"]]
    orig_right  = [e for e in entries if e["orig_correct"]]

    # Among wrong: how many does +4 steering fix?
    wrong_steer_helps  = [e for e in orig_wrong if e["label_pos4"] == 1]
    wrong_steer_harms  = [e for e in orig_wrong if e["label_pos4"] == -1]
    wrong_no_change    = [e for e in orig_wrong if e["label_pos4"] == 0]

    # Among correct: how many does +4 steering break?
    right_steer_helps  = [e for e in orig_right if e["label_pos4"] == 1]   # should be 0 by definition
    right_steer_harms  = [e for e in orig_right if e["label_pos4"] == -1]
    right_no_change    = [e for e in orig_right if e["label_pos4"] == 0]

    print(f"\n{'='*60}")
    print(f"  Total samples : {total}")
    print(f"  orig_correct=1 (right) : {len(orig_right):>7}  ({100*len(orig_right)/total:.1f}%)")
    print(f"  orig_correct=0 (wrong) : {len(orig_wrong):>7}  ({100*len(orig_wrong)/total:.1f}%)")
    print(f"{'='*60}")

    print(f"\n--- Among WRONG samples ({len(orig_wrong)}) ---")
    print(f"  +4 steering fixes  (wrong→correct) : {len(wrong_steer_helps):>6}  ({100*len(wrong_steer_helps)/len(orig_wrong):.1f}%)")
    print(f"  +4 steering harms  (wrong→wrong)   : {len(wrong_steer_harms):>6}  ({100*len(wrong_steer_harms)/len(orig_wrong):.1f}%)")
    print(f"  +4 no change                        : {len(wrong_no_change):>6}  ({100*len(wrong_no_change)/len(orig_wrong):.1f}%)")

    print(f"\n--- Among CORRECT samples ({len(orig_right)}) ---")
    print(f"  +4 steering fixes  (shouldn't exist): {len(right_steer_helps):>6}")
    print(f"  +4 steering harms  (correct→wrong)  : {len(right_steer_harms):>6}  ({100*len(right_steer_harms)/len(orig_right):.1f}%)")
    print(f"  +4 no change                         : {len(right_no_change):>6}  ({100*len(right_no_change)/len(orig_right):.1f}%)")

    print(f"\n--- Simulation: steer ALL wrong samples ---")
    gain = len(wrong_steer_helps)
    loss_from_wrong = len(wrong_steer_harms)  # wrong stays wrong (no actual loss in accuracy)
    print(f"  Accuracy gain  (+correct) : +{gain}")
    print(f"  Accuracy unchanged        : (wrong→wrong counts as no change)")
    print(f"  Net accuracy change       : +{gain}/{total} = +{100*gain/total:.2f}%")

    print(f"\n--- Simulation: steer ALL samples (wrong + correct) ---")
    net = len(wrong_steer_helps) - len(right_steer_harms)
    print(f"  +4 fixes wrong  : +{len(wrong_steer_helps)}")
    print(f"  +4 breaks right : -{len(right_steer_harms)}")
    print(f"  Net change      : {net:+d} / {total} = {100*net/total:+.2f}%")

    print(f"\n--- Key ratios ---")
    steer_rate_overall = len(wrong_steer_helps) / total
    steer_rate_among_wrong = len(wrong_steer_helps) / len(orig_wrong)
    print(f"  Steering effective rate (overall)          : {100*steer_rate_overall:.1f}%")
    print(f"  Steering effective rate (among wrong only) : {100*steer_rate_among_wrong:.1f}%")
    print(f"  Lift from targeting wrong samples          : {steer_rate_among_wrong/steer_rate_overall:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--roles", nargs="*", default=None,
                        help="Filter by roles (default: all)")
    args = parser.parse_args()

    print(f"Loading labels for {args.model}...")
    entries = load_all_entries(args.model, roles_filter=set(args.roles) if args.roles else None)
    print(f"  {len(entries)} entries loaded")

    analyze(entries)


if __name__ == "__main__":
    main()

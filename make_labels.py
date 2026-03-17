"""
make_labels.py — Generate steering labels for ConfSteer project.

For each sample:
  x = hidden state (aligned by index to orig)
  label_pos4: result of +4 alpha steering
  label_neg4: result of -4 alpha steering

Label encoding:
  +1  = steering worked as intended (wrong→correct for +4; correct→wrong for -4)
  -1  = backfire (correct→wrong for +4; wrong→correct for -4)
   0  = no change
"""

import json
import argparse
from pathlib import Path


# ==================== Label Logic ====================

def get_correct_letter_standard(sample):
    """For standard MC tasks: label is int index, convert to letter."""
    return chr(65 + sample["label"])


def get_correct_letter_tqa_mc1(sample):
    """TQA MC1: gold_indices[0] is the correct answer index."""
    return chr(65 + sample["gold_indices"][0])


def get_correct_letter_tqa_mc2(sample):
    """TQA MC2: any option where labels[i]==1 is correct.
    Returns a set of correct letters."""
    return {chr(65 + i) for i, lbl in enumerate(sample["labels"]) if lbl == 1}


def compute_label_pos4(orig_correct, steered_correct):
    """
    +4 steering:
      wrong → correct = +1 (worked)
      correct → wrong = -1 (backfire)
      no change       =  0
    """
    if not orig_correct and steered_correct:
        return 1
    elif orig_correct and not steered_correct:
        return -1
    else:
        return 0


def compute_label_neg4(orig_correct, steered_correct):
    """
    -4 steering:
      correct → wrong = -1 (worked)
      wrong → correct = +1 (backfire)
      no change       =  0
    """
    if orig_correct and not steered_correct:
        return -1
    elif not orig_correct and steered_correct:
        return 1
    else:
        return 0


def is_correct_standard(answer, sample):
    return answer == get_correct_letter_standard(sample)


def is_correct_tqa_mc1(answer, sample):
    return answer == get_correct_letter_tqa_mc1(sample)


def is_correct_tqa_mc2(answer, sample):
    return answer in get_correct_letter_tqa_mc2(sample)


# ==================== File Matching ====================

def find_mdf_file(orig_path: Path, mdf_dir: Path, params: str):
    """
    Find the corresponding mdf file for an orig file.
    Convention: {orig_stem}_{params}.json
    Falls back to searching by common keywords if exact match fails.
    """
    stem = orig_path.stem  # e.g. "AR-LSAT-train_8B_answers"
    target = mdf_dir / f"{stem}_{params}.json"
    if target.exists():
        return target

    # fallback: search by stem prefix
    candidates = list(mdf_dir.glob(f"{stem}_*.json"))
    if len(candidates) == 1:
        return candidates[0]

    # fallback for TQA: orig has "llama3_8B_mc1" but mdf has "8B_answers_mc1"
    # Try matching by size token (e.g. "8B") and mode token (e.g. "mc1")
    import re
    size_match = re.search(r"(\d+B)", stem)
    mode_match = re.search(r"(mc[12])", stem, re.IGNORECASE)
    base_match = re.match(r"(TruthfulQA[_\s]*MC[12])", stem, re.IGNORECASE)
    if size_match and base_match:
        size = size_match.group(1)
        pattern = f"*{size}*{params}.json"
        if mode_match:
            mode = mode_match.group(1).lower()
            pattern = f"*{size}*{mode}*{params}.json"
        candidates = list(mdf_dir.glob(pattern))
        if len(candidates) == 1:
            return candidates[0]

    raise FileNotFoundError(
        f"Cannot find mdf file for '{orig_path.name}' in '{mdf_dir}' with params '{params}'. "
        f"Candidates: {[c.name for c in list(mdf_dir.glob('*.json'))[:10]]}"
    )


# ==================== Core Processing ====================

def process_task(orig_dir: Path, mdf_pos_dir: Path, mdf_neg_dir: Path,
                 out_dir: Path, params: str, task_type: str):
    """Process all orig files in a task directory."""
    orig_files = sorted(orig_dir.glob("*.json"))
    if not orig_files:
        print(f"  [skip] No JSON files in {orig_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for orig_path in orig_files:
        try:
            mdf_pos_path = find_mdf_file(orig_path, mdf_pos_dir, params)
            mdf_neg_path = find_mdf_file(orig_path, mdf_neg_dir, params)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            continue

        with open(orig_path) as f:
            orig_data = json.load(f)["data"]
        with open(mdf_pos_path) as f:
            mdf_pos_data = json.load(f)["data"]
        with open(mdf_neg_path) as f:
            mdf_neg_data = json.load(f)["data"]

        # Build text → index map from orig for alignment verification
        text_to_idx = {s["text"]: i for i, s in enumerate(orig_data)}

        results = []
        mismatch_count = 0

        for i, (orig_s, pos_s, neg_s) in enumerate(
                zip(orig_data, mdf_pos_data, mdf_neg_data)):

            # Verify alignment via text field
            if orig_s["text"] != pos_s["text"] or orig_s["text"] != neg_s["text"]:
                mismatch_count += 1
                # Try to find correct index by text
                if pos_s["text"] in text_to_idx:
                    i = text_to_idx[pos_s["text"]]
                    orig_s = orig_data[i]

            answer_orig = orig_s.get("answer_neutral", "")
            answer_pos4 = pos_s.get("answer_neutral", "")
            answer_neg4 = neg_s.get("answer_neutral", "")

            # Determine correctness based on task type
            if task_type == "mc1":
                orig_correct = is_correct_tqa_mc1(answer_orig, orig_s)
                pos_correct  = is_correct_tqa_mc1(answer_pos4, orig_s)
                neg_correct  = is_correct_tqa_mc1(answer_neg4, orig_s)
                true_label   = get_correct_letter_tqa_mc1(orig_s)
            elif task_type == "mc2":
                orig_correct = is_correct_tqa_mc2(answer_orig, orig_s)
                pos_correct  = is_correct_tqa_mc2(answer_pos4, orig_s)
                neg_correct  = is_correct_tqa_mc2(answer_neg4, orig_s)
                true_label   = ",".join(sorted(get_correct_letter_tqa_mc2(orig_s)))
            else:
                orig_correct = is_correct_standard(answer_orig, orig_s)
                pos_correct  = is_correct_standard(answer_pos4, orig_s)
                neg_correct  = is_correct_standard(answer_neg4, orig_s)
                true_label   = get_correct_letter_standard(orig_s)

            label_pos4 = compute_label_pos4(orig_correct, pos_correct)
            label_neg4 = compute_label_neg4(orig_correct, neg_correct)

            results.append({
                "index":        i,
                "text":         orig_s["text"],
                "true_label":   true_label,
                "answer_orig":  answer_orig,
                "answer_pos4":  answer_pos4,
                "answer_neg4":  answer_neg4,
                "orig_correct": orig_correct,
                "label_pos4":   label_pos4,
                "label_neg4":   label_neg4,
            })

        if mismatch_count > 0:
            print(f"  [warn] {orig_path.name}: {mismatch_count} text mismatches (re-aligned by text)")

        # Statistics
        pos4_counts = {1: 0, 0: 0, -1: 0}
        neg4_counts = {1: 0, 0: 0, -1: 0}
        for r in results:
            pos4_counts[r["label_pos4"]] += 1
            neg4_counts[r["label_neg4"]] += 1

        output = {
            "meta": {
                "orig_file":   orig_path.name,
                "mdf_pos_file": mdf_pos_path.name,
                "mdf_neg_file": mdf_neg_path.name,
                "params":      params,
                "task_type":   task_type,
                "total":       len(results),
                "pos4_stats":  pos4_counts,
                "neg4_stats":  neg4_counts,
            },
            "data": results,
        }

        out_file = out_dir / f"labels_{orig_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  [done] {orig_path.name} → {out_file.name}")
        print(f"         +4: worked={pos4_counts[1]}, no_change={pos4_counts[0]}, backfire={pos4_counts[-1]}")
        print(f"         -4: worked={neg4_counts[-1]}, no_change={neg4_counts[0]}, backfire={neg4_counts[1]}")


# ==================== Task Config ====================

TASK_CONFIGS = {
    "arlsat":  {"type": "standard"},
    "factor":  {"type": "standard"},
    "gpqa":    {"type": "standard"},
    "logiqa":  {"type": "standard"},
    "mmlupro": {"type": "standard"},
    "tqa_mc1": {"type": "mc1",  "task_dir": "tqa"},
    "tqa_mc2": {"type": "mc2",  "task_dir": "tqa"},
}

TQA_MC1_ORIG_PATTERN = "*MC1*"
TQA_MC2_ORIG_PATTERN = "*MC2*"


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Generate steering labels for ConfSteer")
    parser.add_argument("--base_dir", default="/Users/paveenhuang/Downloads/ConfSteer",
                        help="Root ConfSteer directory")
    parser.add_argument("--model", default="llama3", help="Model name (llama3 / qwen3)")
    parser.add_argument("--params", default="20_11_20",
                        help="Steering params suffix: top_start_end (e.g. 20_11_20 or 40_17_30)")
    parser.add_argument("--tasks", nargs="+",
                        default=["arlsat", "factor", "gpqa", "logiqa", "mmlupro", "tqa_mc1", "tqa_mc2"],
                        help="Tasks to process")
    args = parser.parse_args()

    base = Path(args.base_dir)
    model_dir = base / args.model
    out_base = base / "labels" / args.model

    print(f"Model: {args.model}")
    print(f"Params: {args.params}")
    print(f"Tasks: {args.tasks}")
    print()

    for task_key in args.tasks:
        cfg = TASK_CONFIGS.get(task_key)
        if cfg is None:
            print(f"[skip] Unknown task: {task_key}")
            continue

        task_dir_name = cfg.get("task_dir", task_key)
        task_dir = model_dir / task_dir_name
        task_type = cfg["type"]

        print(f"=== {task_key} ===")

        orig_dir    = task_dir / "orig"
        mdf_pos_dir = task_dir / "mdf_4"
        mdf_neg_dir = task_dir / "mdf_-4"
        out_dir     = out_base / task_dir_name

        if not orig_dir.exists():
            print(f"  [skip] {orig_dir} not found")
            continue

        # For TQA, filter orig files by MC1 or MC2
        if task_key in ("tqa_mc1", "tqa_mc2"):
            pattern = TQA_MC1_ORIG_PATTERN if task_key == "tqa_mc1" else TQA_MC2_ORIG_PATTERN
            orig_files = sorted(orig_dir.glob(f"{pattern}.json"))
            if not orig_files:
                print(f"  [skip] No files matching {pattern} in {orig_dir}")
                continue
            # Process each file individually
            for orig_path in orig_files:
                _process_single(orig_path, mdf_pos_dir, mdf_neg_dir,
                                out_dir, args.params, task_type)
        else:
            process_task(orig_dir, mdf_pos_dir, mdf_neg_dir,
                         out_dir, args.params, task_type)

        print()


def _process_single(orig_path, mdf_pos_dir, mdf_neg_dir, out_dir, params, task_type):
    """Process a single orig file (used for TQA split by MC1/MC2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        mdf_pos_path = find_mdf_file(orig_path, mdf_pos_dir, params)
        mdf_neg_path = find_mdf_file(orig_path, mdf_neg_dir, params)
    except FileNotFoundError as e:
        print(f"  [warn] {e}")
        return

    with open(orig_path) as f:
        orig_data = json.load(f)["data"]
    with open(mdf_pos_path) as f:
        mdf_pos_data = json.load(f)["data"]
    with open(mdf_neg_path) as f:
        mdf_neg_data = json.load(f)["data"]

    results = []
    mismatch_count = 0
    text_to_idx = {s["text"]: i for i, s in enumerate(orig_data)}

    for i, (orig_s, pos_s, neg_s) in enumerate(zip(orig_data, mdf_pos_data, mdf_neg_data)):
        if orig_s["text"] != pos_s["text"] or orig_s["text"] != neg_s["text"]:
            mismatch_count += 1
            if pos_s["text"] in text_to_idx:
                i = text_to_idx[pos_s["text"]]
                orig_s = orig_data[i]

        answer_orig = orig_s.get("answer_neutral", "")
        answer_pos4 = pos_s.get("answer_neutral", "")
        answer_neg4 = neg_s.get("answer_neutral", "")

        if task_type == "mc1":
            orig_correct = is_correct_tqa_mc1(answer_orig, orig_s)
            pos_correct  = is_correct_tqa_mc1(answer_pos4, orig_s)
            neg_correct  = is_correct_tqa_mc1(answer_neg4, orig_s)
            true_label   = get_correct_letter_tqa_mc1(orig_s)
        else:
            orig_correct = is_correct_tqa_mc2(answer_orig, orig_s)
            pos_correct  = is_correct_tqa_mc2(answer_pos4, orig_s)
            neg_correct  = is_correct_tqa_mc2(answer_neg4, orig_s)
            true_label   = ",".join(sorted(get_correct_letter_tqa_mc2(orig_s)))

        results.append({
            "index":        i,
            "text":         orig_s["text"],
            "true_label":   true_label,
            "answer_orig":  answer_orig,
            "answer_pos4":  answer_pos4,
            "answer_neg4":  answer_neg4,
            "orig_correct": orig_correct,
            "label_pos4":   compute_label_pos4(orig_correct, pos_correct),
            "label_neg4":   compute_label_neg4(orig_correct, neg_correct),
        })

    if mismatch_count > 0:
        print(f"  [warn] {orig_path.name}: {mismatch_count} text mismatches")

    pos4_counts = {1: 0, 0: 0, -1: 0}
    neg4_counts = {1: 0, 0: 0, -1: 0}
    for r in results:
        pos4_counts[r["label_pos4"]] += 1
        neg4_counts[r["label_neg4"]] += 1

    output = {
        "meta": {
            "orig_file":    orig_path.name,
            "mdf_pos_file": mdf_pos_path.name,
            "mdf_neg_file": mdf_neg_path.name,
            "params":       params,
            "task_type":    task_type,
            "total":        len(results),
            "pos4_stats":   pos4_counts,
            "neg4_stats":   neg4_counts,
        },
        "data": results,
    }

    out_file = out_dir / f"labels_{orig_path.stem}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  [done] {orig_path.name} → {out_file.name}")
    print(f"         +4: worked={pos4_counts[1]}, no_change={pos4_counts[0]}, backfire={pos4_counts[-1]}")
    print(f"         -4: worked={neg4_counts[-1]}, no_change={neg4_counts[0]}, backfire={neg4_counts[1]}")


if __name__ == "__main__":
    main()
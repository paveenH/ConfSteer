"""
make_labels.py — Generate steering labels for ConfSteer project.

For each sample × role:
  x = hidden state (aligned by index to orig)
  label_pos4: result of +4 alpha steering
  label_neg4: result of -4 alpha steering

Label encoding:
  +1  = steering worked as intended (wrong→correct for +4; correct→wrong for -4)
  -1  = backfire (correct→wrong for +4; wrong→correct for -4)
   0  = no change

Each entry in output["data"] has a "role" field.
Total samples = N_questions × N_roles_available.
"""

import json
import re
import argparse
from pathlib import Path


# ==================== Roles ====================
# Edit this list to control which roles are included.
# mmlupro task-specific roles (e.g. biology_expert) are resolved automatically.
GENERIC_ROLES = [
    "neutral",
    "confident",
    "unconfident",
    "expert",
    "non_expert",
    "student",
    "person",
]

# For mmlupro, expert/non_expert/student are task-specific (e.g. biology_expert).
# These are the base names that get a task prefix.
MMLUPRO_TASK_SPECIFIC = {"expert", "non_expert", "student"}


# ==================== Label Logic ====================

def get_correct_letter_standard(sample):
    return chr(65 + sample["label"])


def get_correct_letter_tqa_mc1(sample):
    return chr(65 + sample["gold_indices"][0])


def get_correct_letter_tqa_mc2(sample):
    return {chr(65 + i) for i, lbl in enumerate(sample["labels"]) if lbl == 1}


def compute_label_pos4(orig_correct, steered_correct):
    if not orig_correct and steered_correct:
        return 1
    elif orig_correct and not steered_correct:
        return -1
    return 0


def compute_label_neg4(orig_correct, steered_correct):
    if orig_correct and not steered_correct:
        return -1
    elif not orig_correct and steered_correct:
        return 1
    return 0


def is_correct_standard(answer, sample):
    return answer == get_correct_letter_standard(sample)


def is_correct_tqa_mc1(answer, sample):
    return answer == get_correct_letter_tqa_mc1(sample)


def is_correct_tqa_mc2(answer, sample):
    return answer in get_correct_letter_tqa_mc2(sample)


# ==================== Role Key Resolution ====================

def resolve_role_keys(sample_keys, task_name=None):
    """
    Returns list of (role_label, answer_key) tuples present in sample_keys.

    For mmlupro, task-specific roles are auto-detected from key names.
    For all tasks, generic roles (neutral, confident, etc.) are checked directly.
    """
    answer_keys = {k for k in sample_keys if k.startswith("answer_")}
    resolved = []

    for role in GENERIC_ROLES:
        key = f"answer_{role}"
        if task_name == "mmlupro" and role in MMLUPRO_TASK_SPECIFIC:
            # Find the task-specific variant, e.g. answer_biology_expert
            pattern = re.compile(rf"^answer_(.+_{re.escape(role)})$")
            matches = [k for k in answer_keys if pattern.match(k)]
            for k in matches:
                role_label = pattern.match(k).group(1)  # e.g. "biology_expert"
                resolved.append((role_label, k))
        else:
            if key in answer_keys:
                resolved.append((role, key))

    return resolved


# ==================== File Matching ====================

def find_mdf_file(orig_path: Path, mdf_dir: Path, params: str):
    stem = orig_path.stem
    target = mdf_dir / f"{stem}_{params}.json"
    if target.exists():
        return target

    candidates = list(mdf_dir.glob(f"{stem}_*.json"))
    if len(candidates) == 1:
        return candidates[0]

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

def build_results(orig_data, mdf_pos_data, mdf_neg_data, task_type, task_name=None):
    """
    Build flat results list: one entry per (question, role).
    """
    text_to_idx = {s["text"]: i for i, s in enumerate(orig_data)}
    results = []
    mismatch_count = 0

    if task_type == "standard":
        get_true_label = get_correct_letter_standard
        is_correct     = is_correct_standard
    elif task_type == "mc1":
        get_true_label = get_correct_letter_tqa_mc1
        is_correct     = is_correct_tqa_mc1
    else:  # mc2
        get_true_label = lambda s: ",".join(sorted(get_correct_letter_tqa_mc2(s)))
        is_correct     = is_correct_tqa_mc2

    for i, (orig_s, pos_s, neg_s) in enumerate(
            zip(orig_data, mdf_pos_data, mdf_neg_data)):

        if orig_s["text"] != pos_s["text"] or orig_s["text"] != neg_s["text"]:
            mismatch_count += 1
            if pos_s["text"] in text_to_idx:
                i = text_to_idx[pos_s["text"]]
                orig_s = orig_data[i]

        true_label = get_true_label(orig_s)

        # Resolve available roles from this sample's keys
        role_keys = resolve_role_keys(orig_s.keys(), task_name=task_name)

        for role_label, ans_key in role_keys:
            answer_orig = orig_s.get(ans_key, "")
            answer_pos4 = pos_s.get(ans_key, "")
            answer_neg4 = neg_s.get(ans_key, "")

            if not answer_orig:
                continue  # role missing in this file, skip

            orig_correct = is_correct(answer_orig, orig_s)
            pos_correct  = is_correct(answer_pos4, orig_s) if answer_pos4 else orig_correct
            neg_correct  = is_correct(answer_neg4, orig_s) if answer_neg4 else orig_correct

            results.append({
                "index":        i,
                "role":         role_label,
                "text":         orig_s["text"],
                "true_label":   true_label,
                "answer_orig":  answer_orig,
                "answer_pos4":  answer_pos4,
                "answer_neg4":  answer_neg4,
                "orig_correct": orig_correct,
                "label_pos4":   compute_label_pos4(orig_correct, pos_correct),
                "label_neg4":   compute_label_neg4(orig_correct, neg_correct),
            })

    return results, mismatch_count


def make_stats(results):
    pos4 = {1: 0, 0: 0, -1: 0}
    neg4 = {1: 0, 0: 0, -1: 0}
    roles_seen = set()
    for r in results:
        pos4[r["label_pos4"]] += 1
        neg4[r["label_neg4"]] += 1
        roles_seen.add(r["role"])
    return pos4, neg4, sorted(roles_seen)


def process_file(orig_path, mdf_pos_path, mdf_neg_path, out_dir, params, task_type, task_name=None):
    with open(orig_path) as f:
        orig_data = json.load(f)["data"]
    with open(mdf_pos_path) as f:
        mdf_pos_data = json.load(f)["data"]
    with open(mdf_neg_path) as f:
        mdf_neg_data = json.load(f)["data"]

    results, mismatch_count = build_results(
        orig_data, mdf_pos_data, mdf_neg_data, task_type, task_name=task_name)

    if mismatch_count > 0:
        print(f"  [warn] {orig_path.name}: {mismatch_count} text mismatches (re-aligned by text)")

    pos4_counts, neg4_counts, roles_seen = make_stats(results)

    output = {
        "meta": {
            "orig_file":    orig_path.name,
            "mdf_pos_file": mdf_pos_path.name,
            "mdf_neg_file": mdf_neg_path.name,
            "params":       params,
            "task_type":    task_type,
            "roles":        roles_seen,
            "total":        len(results),
            "pos4_stats":   pos4_counts,
            "neg4_stats":   neg4_counts,
        },
        "data": results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"labels_{orig_path.stem}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  [done] {orig_path.name} → {out_file.name}  (roles: {roles_seen}, entries: {len(results)})")
    print(f"         +4: worked={pos4_counts[1]}, no_change={pos4_counts[0]}, backfire={pos4_counts[-1]}")
    print(f"         -4: worked={neg4_counts[-1]}, no_change={neg4_counts[0]}, backfire={neg4_counts[1]}")


def process_task(orig_dir, mdf_pos_dir, mdf_neg_dir, out_dir, params, task_type, task_name=None):
    orig_files = sorted(orig_dir.glob("*.json"))
    if not orig_files:
        print(f"  [skip] No JSON files in {orig_dir}")
        return
    for orig_path in orig_files:
        try:
            mdf_pos_path = find_mdf_file(orig_path, mdf_pos_dir, params)
            mdf_neg_path = find_mdf_file(orig_path, mdf_neg_dir, params)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            continue
        process_file(orig_path, mdf_pos_path, mdf_neg_path, out_dir, params, task_type, task_name=task_name)


# ==================== Task Config ====================

TASK_CONFIGS = {
    "arlsat":  {"type": "standard"},
    "factor":  {"type": "standard"},
    "gpqa":    {"type": "standard"},
    "logiqa":  {"type": "standard"},
    "mmlupro": {"type": "standard"},
    "tqa_mc1": {"type": "mc1", "task_dir": "tqa"},
    "tqa_mc2": {"type": "mc2", "task_dir": "tqa"},
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
                        help="Steering params suffix (e.g. 20_11_20 or 20_17_26)")
    parser.add_argument("--tasks", nargs="+",
                        default=["arlsat", "factor", "gpqa", "logiqa", "mmlupro", "tqa_mc1", "tqa_mc2"],
                        help="Tasks to process")
    args = parser.parse_args()

    base     = Path(args.base_dir)
    model_dir = base / "answer" / args.model
    out_base  = base / "labels" / args.model

    print(f"Model:  {args.model}")
    print(f"Params: {args.params}")
    print(f"Tasks:  {args.tasks}")
    print(f"Roles:  {GENERIC_ROLES}")
    print()

    for task_key in args.tasks:
        cfg = TASK_CONFIGS.get(task_key)
        if cfg is None:
            print(f"[skip] Unknown task: {task_key}")
            continue

        task_dir_name = cfg.get("task_dir", task_key)
        task_dir  = model_dir / task_dir_name
        task_type = cfg["type"]
        task_name = task_key  # used for mmlupro role resolution

        print(f"=== {task_key} ===")

        orig_dir    = task_dir / "orig"
        mdf_pos_dir = task_dir / "mdf_4"
        mdf_neg_dir = task_dir / "mdf_-4"
        out_dir     = out_base / task_dir_name

        if not orig_dir.exists():
            print(f"  [skip] {orig_dir} not found")
            continue

        if task_key in ("tqa_mc1", "tqa_mc2"):
            pattern = TQA_MC1_ORIG_PATTERN if task_key == "tqa_mc1" else TQA_MC2_ORIG_PATTERN
            orig_files = sorted(orig_dir.glob(f"{pattern}.json"))
            if not orig_files:
                print(f"  [skip] No files matching {pattern} in {orig_dir}")
                continue
            for orig_path in orig_files:
                try:
                    mdf_pos_path = find_mdf_file(orig_path, mdf_pos_dir, args.params)
                    mdf_neg_path = find_mdf_file(orig_path, mdf_neg_dir, args.params)
                except FileNotFoundError as e:
                    print(f"  [warn] {e}")
                    continue
                process_file(orig_path, mdf_pos_path, mdf_neg_path,
                             out_dir, args.params, task_type, task_name=task_key)
        else:
            process_task(orig_dir, mdf_pos_dir, mdf_neg_dir,
                         out_dir, args.params, task_type, task_name=task_name)

        print()


if __name__ == "__main__":
    main()

"""
eval_classifier_mmlupro.py — OOD evaluation of MMLU-trained classifier on MMLU-Pro HS
=======================================================================================
Loads the MMLU-trained PCA-CNN classifier and evaluates it on MMLU-Pro hidden states
to measure OOD generalization (AUC, accuracy, per-class recall).

Usage:
  python eval_classifier_mmlupro.py \
    --model llama3 \
    --size 8B \
    --clf_dir models/llama3_pca128_mmlu_cnn_k7 \
    --role neutral

Data layout assumed:
  answer/llama3/mmlupro/orig/{task}_8B_answers.json   — contains answer_neutral, label (int index)
  HiddenStates/llama3/mmlupro/neutral_{task}_8B.h5    — hidden states
"""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

BASE_DIR   = Path(__file__).parent
HIDDEN_DIR = BASE_DIR / "HiddenStates"
ANSWER_DIR = BASE_DIR / "answer"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ──────────────────────────────────────────────
#  PCA-CNN with residual (must match trained model)
# ──────────────────────────────────────────────

class PCA_CNN(nn.Module):
    def __init__(self, num_layers, pca_dim, cnn_channels=64, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(pca_dim, cnn_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2)
        self.proj  = nn.Conv1d(pca_dim, cnn_channels, 1) if pca_dim != cnn_channels else nn.Identity()
        self.act   = nn.GELU()
        self.attn  = nn.Linear(cnn_channels, 1)
        self.head  = nn.Sequential(
            nn.Linear(cnn_channels, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        residual = self.proj(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x) + residual)
        x = x.permute(0, 2, 1)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        x = (x * w).sum(dim=1)
        return self.head(x)


# ──────────────────────────────────────────────
#  Load classifier
# ──────────────────────────────────────────────

def load_classifier(clf_dir: Path, device: torch.device):
    ckpt = torch.load(clf_dir / "model.pt", map_location="cpu")
    cfg  = ckpt["config"]
    model = PCA_CNN(
        num_layers=cfg["num_layers"],
        pca_dim=cfg["pca_dim"],
        cnn_channels=cfg.get("cnn_channels", 64),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(clf_dir / "preprocessor.pkl", "rb") as f:
        prep = pickle.load(f)

    print(f"  Classifier: L={cfg['num_layers']}, pca_dim={cfg['pca_dim']}, "
          f"cnn_ch={cfg.get('cnn_channels',64)}, kernel={cfg.get('kernel_size',3)}")
    return model, prep["scalers"], prep["pcas"], cfg["pca_dim"]


# ──────────────────────────────────────────────
#  Per-layer scaler + PCA transform
# ──────────────────────────────────────────────

def transform(X: np.ndarray, scalers, pcas, pca_dim: int) -> np.ndarray:
    N, L, _ = X.shape
    out = np.zeros((N, L, pca_dim), dtype=np.float32)
    for l in range(L):
        X_l = scalers[l].transform(X[:, l, :])
        n_comp = pcas[l].n_components_
        out[:, l, :n_comp] = pcas[l].transform(X_l)
    return out


# ──────────────────────────────────────────────
#  Role → answer key / H5 prefix
# ──────────────────────────────────────────────

def role_to_answer_key(role: str, task: str) -> str:
    task_slug = task.replace(" ", "_")
    r = role.replace("{task}", task_slug)
    return "answer_" + r.replace(" ", "_")


def role_to_h5_prefix(role: str, task: str) -> str:
    task_slug = task.replace(" ", "_")
    r = role.replace("{task}", task_slug)
    return r.replace(" ", "_")


# ──────────────────────────────────────────────
#  Extract samples from MMLU-Pro
# ──────────────────────────────────────────────

def extract(model: str, size: str, role: str):
    ans_dir = ANSWER_DIR / model / "mmlupro" / "orig"
    hs_dir  = HIDDEN_DIR / model / "mmlupro"

    ans_files = sorted(ans_dir.glob(f"*_{size}_answers.json"))
    if not ans_files:
        raise FileNotFoundError(f"No answer files found in {ans_dir}")
    print(f"  Found {len(ans_files)} answer files")

    X_list, y_list, task_list = [], [], []
    skipped = []

    for ans_path in ans_files:
        with open(ans_path, encoding="utf-8") as f:
            d = json.load(f)
        samples = d["data"]
        if not samples:
            continue

        task      = samples[0]["task"]
        task_slug = task.replace(" ", "_")
        ans_key   = role_to_answer_key(role, task)
        h5_prefix = role_to_h5_prefix(role, task)
        h5_path   = hs_dir / f"{h5_prefix}_{task_slug}_{size}.h5"

        if not h5_path.exists():
            skipped.append(h5_path.name)
            continue

        with h5py.File(h5_path, "r") as hf:
            hs    = hf["hidden_states"]   # (N, n_layers, hidden_dim)
            n_h5  = hs.shape[0]

            for idx, sample in enumerate(samples):
                if idx >= n_h5:
                    break
                pred = sample.get(ans_key)
                if pred is None:
                    continue
                # MMLU-Pro: label is integer index, true letter = LETTERS[label]
                true_letter  = LETTERS[int(sample["label"])]
                orig_correct = int(pred == true_letter)
                X_list.append(hs[idx, :, :])
                y_list.append(orig_correct)
                task_list.append(task_slug)

    if skipped:
        print(f"  Skipped {len(skipped)} H5 files not found")

    if not X_list:
        raise RuntimeError("No samples extracted — check paths and role name.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int64)
    print(f"  Extracted {len(X)} samples, shape: {X.shape}")
    print(f"  y — correct(1): {(y==1).sum()}  wrong(0): {(y==0).sum()}  "
          f"({(y==1).mean()*100:.1f}% correct)")
    return X, y, task_list


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="llama3")
    parser.add_argument("--size",     default="8B")
    parser.add_argument("--role",     default="neutral")
    parser.add_argument("--clf_dir",  required=True,
                        help="Path to classifier dir (model.pt + preprocessor.pkl)")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch",    type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    clf_dir = Path(args.clf_dir)

    print(f"\n{'='*55}")
    print(f"  OOD Eval: MMLU-trained classifier → MMLU-Pro HS")
    print(f"  Model  : {args.model}-{args.size}")
    print(f"  Role   : {args.role}")
    print(f"  Clf    : {clf_dir}")
    print(f"{'='*55}\n")

    print("[1] Loading classifier...")
    clf_model, scalers, pcas, pca_dim = load_classifier(clf_dir, device)

    print("\n[2] Extracting MMLU-Pro hidden states...")
    X, y, tasks = extract(args.model, args.size, args.role)

    print("\n[3] Applying per-layer scaler + PCA...")
    X_pca = transform(X, scalers, pcas, pca_dim)
    print(f"  Transformed: {X_pca.shape}")

    print("\n[4] Running inference...")
    clf_model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(X_pca), args.batch):
            xb = torch.tensor(X_pca[i:i+args.batch]).to(device)
            logits = clf_model(xb)
            preds  = logits.argmax(1).cpu().tolist()
            probs  = torch.softmax(logits, 1)[:, 1].cpu().tolist()
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(y, all_probs)
    acc = (all_preds == y).mean() * 100

    print(f"\n[5] Results:")
    print(f"  Overall accuracy : {acc:.2f}%")
    print(f"  ROC-AUC          : {auc:.3f}")
    print(f"\n  Predicted steer rate (y_pred=0): {(all_preds==0).mean()*100:.1f}%")
    print(f"\n  Classification report:")
    print(classification_report(y, all_preds, target_names=["wrong(0)", "correct(1)"]))
    print("  Confusion matrix (rows=true, cols=pred):  labels: 0, 1")
    print(confusion_matrix(y, all_preds, labels=[0, 1]))

    # Per-task breakdown
    print(f"\n  Per-task AUC (top 10 best / worst):")
    task_names = sorted(set(tasks))
    task_aucs  = []
    for t in task_names:
        idx = [i for i, tt in enumerate(tasks) if tt == t]
        if len(set(y[idx])) < 2:
            continue
        t_auc = roc_auc_score(y[idx], all_probs[idx])
        t_acc = (all_preds[idx] == y[idx]).mean() * 100
        t_sr  = (all_preds[idx] == 0).mean() * 100
        task_aucs.append((t, t_auc, t_acc, t_sr, len(idx)))

    task_aucs.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Task':<45} {'AUC':>6} {'Acc':>7} {'SteerRate':>10} {'N':>5}")
    print(f"  {'-'*75}")
    for row in task_aucs[:10]:
        print(f"  {row[0]:<45} {row[1]:>6.3f} {row[2]:>6.1f}% {row[3]:>9.1f}% {row[4]:>5}")
    print("  ...")
    for row in task_aucs[-10:]:
        print(f"  {row[0]:<45} {row[1]:>6.3f} {row[2]:>6.1f}% {row[3]:>9.1f}% {row[4]:>5}")

    print(f"\n  Mean per-task AUC: {np.mean([r[1] for r in task_aucs]):.3f}")
    print(f"  Mean steer rate  : {np.mean([r[3] for r in task_aucs]):.1f}%")


if __name__ == "__main__":
    main()

"""
classifier_sparse_attn.py — Sparse Attention over hidden dimensions
=====================================================================
For each hidden dimension d, track its trajectory across L layers,
then use sparse (top-k) attention to select the most informative dims.

Architecture:
  Input (N, L, D)
    → Per-layer StandardScaler (fit on train)
    → Transpose to (N, D, L)           [each dim = a sequence across layers]
    → Per-dim Linear projection: L → proj_dim
    → Sparse Attention: score each dim → keep top-k → weighted sum
    → MLP head → 2 classes

Key idea: directly models "which hidden dimensions are most predictive
of correctness across all layers", aligning with the intuition that
certain indices (e.g. dim 2004) carry consistent cross-layer signal.

Usage:
  python classifier_sparse_attn.py --model llama3 \
    --train samples/llama3/samples_orig_all_5k_train.npz \
    --test  samples/llama3/samples_orig_all_5k_test.npz \
    --topk 512
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR    = Path(__file__).parent
RANDOM_SEED = 42


# ==================== Data loading ====================

def load_samples(path: Path):
    data  = np.load(path, allow_pickle=False)
    X     = data["X"]
    y     = data["y"]
    roles = list(data["roles"])
    print(f"  Loaded: shape={X.shape}, roles={roles}")
    print(f"  y — 1: {(y==1).sum()}, 0: {(y==0).sum()}")
    return X.astype(np.float32), y.astype(np.int64)


def scale_per_layer(X_tr, X_te):
    _, L, _ = X_tr.shape
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    for l in range(L):
        sc = StandardScaler()
        X_tr[:, l, :] = sc.fit_transform(X_tr[:, l, :])
        X_te[:, l, :] = sc.transform(X_te[:, l, :])
    return X_tr, X_te


# ==================== Model ====================

class SparseDimAttention(nn.Module):
    """
    Sparse attention over hidden dimensions.

    Steps:
      1. Transpose input to (B, D, L) — each dim is a sequence across layers
      2. Project each dim's layer-trajectory: L → proj_dim
      3. Score each dim with a learned linear scorer
      4. Top-k sparse attention: keep only the k highest-scoring dims
      5. Weighted sum → (B, proj_dim)
      6. MLP head → 2 classes

    topk: number of hidden dimensions to attend to (out of D=4096)
    """

    def __init__(self, num_layers: int, hidden_dim: int,
                 proj_dim: int = 64, topk: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.topk     = topk
        self.proj_dim = proj_dim

        # Project each dim's trajectory across layers: (L,) → proj_dim
        # Applied independently per dimension (shared weights)
        self.dim_proj = nn.Linear(num_layers, proj_dim)

        # Score each dimension for attention
        self.scorer = nn.Linear(proj_dim, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x.permute(0, 2, 1)          # (B, D, L)
        h = self.dim_proj(x)             # (B, D, proj_dim)

        # Score each dim
        scores = self.scorer(h).squeeze(-1)   # (B, D)

        # Sparse: keep top-k dims
        topk_scores, topk_idx = torch.topk(scores, self.topk, dim=-1)  # (B, topk)
        topk_weights = torch.softmax(topk_scores, dim=-1).unsqueeze(-1) # (B, topk, 1)

        # Gather top-k dim representations
        # h: (B, D, proj_dim) → gather along dim 1
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, self.proj_dim)  # (B, topk, proj_dim)
        topk_h = torch.gather(h, 1, idx_expanded)                            # (B, topk, proj_dim)

        # Weighted sum
        out = (topk_h * topk_weights).sum(dim=1)  # (B, proj_dim)

        return self.head(out)


# ==================== Train / Eval ====================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds, labels, probs = [], [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss   = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        p = logits.argmax(1)
        correct += (p == yb).sum().item()
        n       += len(yb)
        preds.extend(p.cpu().tolist())
        labels.extend(yb.cpu().tolist())
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
    return total_loss / n, correct / n, preds, labels, probs


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--train",    required=True)
    parser.add_argument("--test",     required=True)
    parser.add_argument("--proj_dim", type=int,   default=64,
                        help="Projection dim per hidden dimension's layer trajectory")
    parser.add_argument("--topk",     type=int,   default=512,
                        help="Number of hidden dims to attend to (out of 4096)")
    parser.add_argument("--dropout",  type=float, default=0.3)
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--device",   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    print(f"\n{'='*55}")
    print(f"  Sparse Dim-Attention Classifier")
    print(f"  Model    : {args.model}")
    print(f"  Train    : {args.train}")
    print(f"  Test     : {args.test}")
    print(f"  top-k    : {args.topk} / 4096 dims")
    print(f"  proj_dim : {args.proj_dim}")
    print(f"  Device   : {device}")
    print(f"{'='*55}\n")

    print("[1] Loading samples...")
    X_tr, y_train = load_samples(Path(args.train))
    X_te, y_test  = load_samples(Path(args.test))
    _, L, D = X_tr.shape

    print(f"\n[2] Per-layer StandardScaler...")
    X_train, X_test = scale_per_layer(X_tr, X_te)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = SparseDimAttention(
        num_layers=L, hidden_dim=D,
        proj_dim=args.proj_dim,
        topk=min(args.topk, D),
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[3] Model: {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[4] Training ({args.epochs} epochs)...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'-'*50}")

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        marker = " ←" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  {val_loss:>8.4f}  {val_acc:>7.3f}{marker}")

    model.load_state_dict(best_state)
    _, _, all_preds, all_labels, all_probs = evaluate(model, test_loader, criterion, device)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n[5] Evaluation (best checkpoint):")
    print(classification_report(all_labels, all_preds, target_names=["wrong(0)", "correct(1)"]))
    print(f"  ROC-AUC: {auc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):  labels: 0, 1")
    print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))


if __name__ == "__main__":
    main()

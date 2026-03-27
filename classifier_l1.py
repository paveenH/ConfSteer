"""
classifier_l1.py — L1-regularized Feature Selection + MLP
===========================================================
Two-stage approach:
  Stage 1 — Feature Selection via L1 (Lasso-style sparse linear layer):
    A single Linear(D_total, 1) with L1 penalty on weights identifies
    which hidden dimensions (across all layers) are most discriminative.
    D_total = L × D (all layer-dim combinations flattened).

  Stage 2 — MLP trained on selected features:
    After Stage 1, keep only the top-k dimensions by |weight| magnitude,
    then train a small MLP on the compressed input.

Why L1?
  L1 (Lasso) penalty drives small weights exactly to zero, performing
  automatic feature selection. Unlike PCA (unsupervised), L1 selects
  dimensions that are directly predictive of the label.

Architecture:
  Input (N, L, D)
    → Per-layer StandardScaler (fit on train)
    → Flatten: (N, L*D)
    → Stage 1: SparseLinear(L*D → 1) with L1 penalty  [feature selection]
    → Select top-k indices by |weight|
    → Stage 2: MLP(k → 256 → 64 → 2)                  [classification]

Usage:
  python classifier_l1.py --model llama3 \\
    --train samples/llama3/samples_orig_all_5k_train.npz \\
    --test  samples/llama3/samples_orig_all_5k_test.npz \\
    --topk 1024 --l1_lambda 1e-4

  # Skip Stage 1, go straight to full MLP (baseline):
  python classifier_l1.py ... --topk 0
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


# ==================== Stage 1: L1 sparse selector ====================

class SparseLinear(nn.Module):
    """Single linear layer trained with L1 penalty for feature selection."""

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)  # (B, 1)

    def l1_penalty(self):
        return self.linear.weight.abs().sum()


def train_sparse_selector(X_flat, y, l1_lambda, epochs, lr, batch_size, device):
    """Train SparseLinear with L1 penalty; returns weight tensor (in_features,)."""
    N, D_total = X_flat.shape
    ds = TensorDataset(torch.tensor(X_flat), torch.tensor(y).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SparseLinear(D_total).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    print(f"  [Stage 1] Training sparse selector ({epochs} epochs, λ={l1_lambda})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(-1)
            loss = bce(logits, yb) + l1_lambda * model.l1_penalty()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += ((logits > 0).float() == yb).sum().item()
            n          += len(yb)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  loss={total_loss/n:.4f}  acc={correct/n:.3f}")

    weights = model.linear.weight.data.squeeze(0).cpu()  # (D_total,)
    n_nonzero = (weights.abs() > 1e-8).sum().item()
    print(f"  [Stage 1] Non-zero weights: {n_nonzero} / {D_total}")
    return weights


def select_top_features(weights, topk, D_total):
    """Return indices of top-k features by |weight|, sorted."""
    if topk <= 0 or topk >= D_total:
        return torch.arange(D_total)
    vals, idx = torch.topk(weights.abs(), topk)
    idx_sorted = idx.sort().values
    frac = weights[idx_sorted].abs().sum() / weights.abs().sum()
    print(f"  [Stage 1] Top-{topk} features cover {100*frac:.1f}% of total |weight|")
    return idx_sorted


# ==================== Stage 2: MLP classifier ====================

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


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
    parser.add_argument("--model",      default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--train",      required=True)
    parser.add_argument("--test",       required=True)
    # Stage 1 options
    parser.add_argument("--topk",       type=int,   default=1024,
                        help="Top-k features to keep after L1 selection (0 = keep all)")
    parser.add_argument("--l1_lambda",  type=float, default=1e-4,
                        help="L1 regularization strength for sparse selector")
    parser.add_argument("--s1_epochs",  type=int,   default=20,
                        help="Epochs for Stage 1 sparse selector")
    parser.add_argument("--s1_lr",      type=float, default=1e-3)
    parser.add_argument("--s1_batch",   type=int,   default=256)
    # Stage 2 options
    parser.add_argument("--hidden",     type=int,   default=256,
                        help="Hidden size of Stage 2 MLP")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--device",     default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  L1 Feature-Selection + MLP Classifier")
    print(f"  Model      : {args.model}")
    print(f"  Train      : {args.train}")
    print(f"  Test       : {args.test}")
    print(f"  Top-k      : {args.topk}  (0 = all)")
    print(f"  L1 lambda  : {args.l1_lambda}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    print("[1] Loading samples...")
    X_tr, y_train = load_samples(Path(args.train))
    X_te, y_test  = load_samples(Path(args.test))
    _, L, D = X_tr.shape
    D_total = L * D

    print(f"\n[2] Per-layer StandardScaler...")
    X_train, X_test = scale_per_layer(X_tr, X_te)

    # Flatten for Stage 1 and MLP
    X_train_flat = X_train.reshape(len(X_train), -1)  # (N_tr, L*D)
    X_test_flat  = X_test.reshape(len(X_test),  -1)   # (N_te, L*D)

    # ---- Stage 1: L1 feature selection ----
    print(f"\n[3] Stage 1 — L1 Sparse Feature Selection (top-{args.topk} of {D_total})...")
    if args.topk > 0:
        weights = train_sparse_selector(
            X_train_flat, y_train,
            l1_lambda=args.l1_lambda,
            epochs=args.s1_epochs,
            lr=args.s1_lr,
            batch_size=args.s1_batch,
            device=device,
        )
        sel_idx = select_top_features(weights, args.topk, D_total)
        sel_idx_np = sel_idx.numpy()

        X_tr_sel = X_train_flat[:, sel_idx_np]
        X_te_sel = X_test_flat[:,  sel_idx_np]
        in_features = len(sel_idx_np)
        print(f"  Selected {in_features} features  (L={L}, D={D})")

        # Report which (layer, dim) the top-20 features come from
        print(f"\n  Top-20 selected (layer, dim_idx):")
        top20 = sel_idx_np[:20]
        for fi in top20:
            layer_i = fi // D
            dim_i   = fi  % D
            w = weights[fi].item()
            print(f"    feature {fi:>7}  →  layer {layer_i:>2}, dim {dim_i:>4}  (w={w:+.4f})")
    else:
        print("  topk=0 → skipping Stage 1, using all features.")
        X_tr_sel  = X_train_flat
        X_te_sel  = X_test_flat
        in_features = D_total

    # ---- Stage 2: MLP ----
    print(f"\n[4] Stage 2 — MLP({in_features} → {args.hidden} → 64 → 2)...")
    train_ds = TensorDataset(torch.tensor(X_tr_sel), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_te_sel), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = MLP(in_features=in_features, hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[5] Training ({args.epochs} epochs)...")
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

    print(f"\n[6] Evaluation (best checkpoint):")
    print(classification_report(all_labels, all_preds, target_names=["wrong(0)", "correct(1)"]))
    print(f"  ROC-AUC: {auc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):  labels: 0, 1")
    print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))


if __name__ == "__main__":
    main()

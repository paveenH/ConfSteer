"""
classifier_l1_cnn.py — Per-layer L1 Feature Selection + 1D-CNN
================================================================
Two-stage approach:

  Stage 1 — Per-layer L1 feature selection:
    For each layer l independently, train a Linear(D → 1) with L1 penalty
    (BCE + λ·|w|₁) to identify the top-d most discriminative hidden dims.
    This produces a (L, d) index map.

  Stage 2 — 1D-CNN classifier:
    Compress input (N, L, D) → (N, L, d) using the selected indices,
    then feed into the same LayerAttentionCNN used in classifier_cnn.py.
    d plays the role of hidden_dim; layer structure is fully preserved.

Why per-layer (vs global L1)?
  Global L1 flattens all L×D features and loses the layer axis.
  Per-layer L1 selects the best dims within each layer independently,
  preserving the (N, L, d) structure that the CNN exploits.

Architecture:
  Input (N, L, D)
    → Per-layer StandardScaler (fit on train)
    → Stage 1: per-layer Linear(D→1) + L1  [feature selection, numpy]
    → Select top-d dims per layer → (N, L, d)
    → Stage 2: LayerAttentionCNN(L, d)     [1D-CNN + layer attention + MLP]

Usage:
  python classifier_l1_cnn.py --model llama3 \\
    --train samples/llama3/samples_orig_all_25k_train.npz \\
    --test  samples/llama3/samples_orig_all_25k_test.npz \\
    --topd 256 --l1_lambda 1e-4
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
    """Fit StandardScaler per layer on train, transform both splits."""
    _, L, _ = X_tr.shape
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    scalers = []
    for l in range(L):
        sc = StandardScaler()
        X_tr[:, l, :] = sc.fit_transform(X_tr[:, l, :])
        X_te[:, l, :]  = sc.transform(X_te[:, l, :])
        scalers.append(sc)
    return X_tr, X_te, scalers


# ==================== Stage 1: per-layer L1 selector ====================

class _SparseLinear(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

    def l1_penalty(self):
        return self.linear.weight.abs().sum()


def _train_layer_selector(X_layer, y, l1_lambda, epochs, lr, batch_size, device):
    """
    Train a sparse linear selector for one layer.
    X_layer: (N, D) float32 numpy array (already scaled)
    Returns weight tensor of shape (D,).
    """
    ds     = TensorDataset(torch.tensor(X_layer), torch.tensor(y).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model  = _SparseLinear(X_layer.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    bce    = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = bce(model(Xb).squeeze(-1), yb) + l1_lambda * model.l1_penalty()
            loss.backward()
            opt.step()

    return model.linear.weight.data.squeeze(0).cpu()  # (D,)


def fit_l1_indices(X_train, y_train, topd, l1_lambda, epochs, lr, batch_size, device):
    """
    Run per-layer L1 selection on training data.
    Returns sel_idx: (L, topd) int64 numpy array — selected dim indices per layer.
    """
    _, L, D = X_train.shape
    topd    = min(topd, D)
    sel_idx = np.zeros((L, topd), dtype=np.int64)

    print(f"  [Stage 1] Per-layer L1 selection: {L} layers × top-{topd} / {D} dims ...")
    for l in range(L):
        w = _train_layer_selector(
            X_train[:, l, :], y_train,
            l1_lambda=l1_lambda, epochs=epochs, lr=lr,
            batch_size=batch_size, device=device,
        )
        _, idx = torch.topk(w.abs(), topd)
        sel_idx[l] = idx.sort().values.numpy()

        if (l + 1) % 5 == 0 or l == 0 or l == L - 1:
            n_nonzero = (w.abs() > 1e-8).sum().item()
            print(f"    Layer {l:>2}: non-zero={n_nonzero}/{D}, "
                  f"top-{topd} |w| sum={w[sel_idx[l]].abs().sum():.3f} "
                  f"/ total={w.abs().sum():.3f}")

    return sel_idx  # (L, topd)


def apply_l1_selection(X, sel_idx):
    """
    Select top-d dims per layer.
    X:       (N, L, D)
    sel_idx: (L, topd)
    Returns: (N, L, topd)
    """
    N, L, _ = X.shape
    topd = sel_idx.shape[1]
    out  = np.zeros((N, L, topd), dtype=np.float32)
    for l in range(L):
        out[:, l, :] = X[:, l, sel_idx[l]]
    return out


# ==================== Stage 2: 1D-CNN classifier ====================

class LayerAttentionCNN(nn.Module):
    """
    1D-CNN over layers + learned layer attention → 2-class output.
    Same architecture as classifier_cnn.py; hidden_dim = topd after L1 selection.
    """
    def __init__(self, num_layers: int, hidden_dim: int,
                 proj_dim: int = 64, cnn_channels: int = 64,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.cnn  = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.attn = nn.Linear(cnn_channels, 1)
        self.head = nn.Sequential(
            nn.Linear(cnn_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.proj(x)                              # (B, L, proj_dim)
        x = x.permute(0, 2, 1)                       # (B, proj_dim, L)
        x = self.cnn(x)                               # (B, cnn_ch, L)
        x = x.permute(0, 2, 1)                       # (B, L, cnn_ch)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        x = (x * w).sum(dim=1)                       # (B, cnn_ch)
        return self.head(x)


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
    parser.add_argument("--model",      default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--train",      required=True)
    parser.add_argument("--test",       required=True)
    # Stage 1 options
    parser.add_argument("--topd",       type=int,   default=256,
                        help="Top-d dims to keep per layer after L1 selection")
    parser.add_argument("--l1_lambda",  type=float, default=1e-4,
                        help="L1 regularization strength")
    parser.add_argument("--s1_epochs",  type=int,   default=10,
                        help="Epochs for per-layer L1 selector")
    parser.add_argument("--s1_lr",      type=float, default=1e-3)
    parser.add_argument("--s1_batch",   type=int,   default=256)
    # Stage 2 options
    parser.add_argument("--proj_dim",     type=int,   default=64)
    parser.add_argument("--cnn_channels", type=int,   default=64)
    parser.add_argument("--kernel_size",  type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--device",   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Per-layer L1 Selection + 1D-CNN Classifier")
    print(f"  Model      : {args.model}")
    print(f"  Train      : {args.train}")
    print(f"  Test       : {args.test}")
    print(f"  Top-d/layer: {args.topd}")
    print(f"  L1 lambda  : {args.l1_lambda}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # ── Load ──
    print("[1] Loading samples...")
    X_tr, y_train = load_samples(Path(args.train))
    X_te, y_test  = load_samples(Path(args.test))
    _, L, D = X_tr.shape

    # ── Scale ──
    print(f"\n[2] Per-layer StandardScaler...")
    X_tr, X_te, _ = scale_per_layer(X_tr, X_te)

    # ── Stage 1: per-layer L1 selection ──
    print(f"\n[3] Stage 1 — Per-layer L1 feature selection (top-{args.topd} / {D})...")
    sel_idx = fit_l1_indices(
        X_tr, y_train,
        topd=args.topd,
        l1_lambda=args.l1_lambda,
        epochs=args.s1_epochs,
        lr=args.s1_lr,
        batch_size=args.s1_batch,
        device=device,
    )

    X_tr_sel = apply_l1_selection(X_tr, sel_idx)  # (N_tr, L, topd)
    X_te_sel = apply_l1_selection(X_te, sel_idx)  # (N_te, L, topd)
    print(f"  Compressed: {X_tr.shape} → {X_tr_sel.shape}")

    # ── Stage 2: CNN ──
    print(f"\n[4] Stage 2 — LayerAttentionCNN({L}, {args.topd})...")
    train_ds = TensorDataset(torch.tensor(X_tr_sel), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_te_sel), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = LayerAttentionCNN(
        num_layers=L, hidden_dim=args.topd,
        proj_dim=args.proj_dim, cnn_channels=args.cnn_channels,
        kernel_size=args.kernel_size, dropout=args.dropout,
    ).to(device)
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

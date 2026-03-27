"""
classifier_cnn2d.py — 2D-CNN over (layer × hidden_dim)
========================================================
Treats the hidden state matrix (L, D) as a 2D image.
2D convolution captures local patterns across both the layer axis
and the hidden-dimension axis simultaneously.

Architecture:
  Input (N, L, D)
    → Per-layer StandardScaler (fit on train)
    → Reshape to (N, 1, L, D)                 [single channel image]
    → 2D-CNN: kernel (3, k_dim)               [layer × dim local patterns]
    → Global Average Pooling over spatial dims
    → MLP head → 2 classes

Usage:
  python classifier_cnn2d.py --model llama3 \
    --train samples/llama3/samples_orig_all_5k_train.npz \
    --test  samples/llama3/samples_orig_all_5k_test.npz
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

class CNN2D(nn.Module):
    """
    2D-CNN treating the hidden state (L, D) as a single-channel image.

    kernel_layer: convolution span across layer axis (default 3)
    kernel_dim:   convolution span across hidden-dim axis (default 64)
                  — captures local patterns within groups of dimensions
    """

    def __init__(self, num_layers: int, hidden_dim: int,
                 channels1: int = 32, channels2: int = 64,
                 kernel_layer: int = 3, kernel_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        pad_l = kernel_layer // 2
        pad_d = kernel_dim   // 2

        self.conv = nn.Sequential(
            # (B, 1, L, D) → (B, ch1, L, D)
            nn.Conv2d(1, channels1,
                      kernel_size=(kernel_layer, kernel_dim),
                      padding=(pad_l, pad_d)),
            nn.GELU(),
            nn.Conv2d(channels1, channels2,
                      kernel_size=(kernel_layer, kernel_dim),
                      padding=(pad_l, pad_d)),
            nn.GELU(),
        )

        # Global average pooling over (L, D) → (B, channels2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(channels2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x.unsqueeze(1)          # (B, 1, L, D)
        x = self.conv(x)            # (B, ch2, L, D)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (B, ch2)
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
    parser.add_argument("--model",    default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--train",    required=True)
    parser.add_argument("--test",     required=True)
    parser.add_argument("--channels1",   type=int,   default=32)
    parser.add_argument("--channels2",   type=int,   default=64)
    parser.add_argument("--kernel_layer",type=int,   default=3,
                        help="Conv kernel size along layer axis")
    parser.add_argument("--kernel_dim",  type=int,   default=64,
                        help="Conv kernel size along hidden-dim axis")
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--batch",   type=int,   default=32,
                        help="Smaller batch recommended for 2D-CNN memory")
    parser.add_argument("--device",  default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    print(f"\n{'='*55}")
    print(f"  2D-CNN Classifier")
    print(f"  Model         : {args.model}")
    print(f"  Train         : {args.train}")
    print(f"  Test          : {args.test}")
    print(f"  Kernel (L×D)  : {args.kernel_layer}×{args.kernel_dim}")
    print(f"  Device        : {device}")
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

    model = CNN2D(
        num_layers=L, hidden_dim=D,
        channels1=args.channels1, channels2=args.channels2,
        kernel_layer=args.kernel_layer, kernel_dim=args.kernel_dim,
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

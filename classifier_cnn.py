"""
ConfSteer CNN Classifier (Binary)
==================================
Binary classification: should we apply +4 steering?

Label definition:
  1 (steer)    : label_pos4 == +1  (wrong→correct under +4 steering)
  0 (no_steer) : all others

Architecture:
  Input (N, L, D)
    → Per-layer StandardScaler
    → Linear projection: D → proj_dim (per layer, shared weights)
    → 1D-CNN over layers (kernel_size=3)
    → Layer Attention: learned weighted sum over layers
    → MLP head → 2 classes

Usage (question-level split, recommended):
  python classifier_cnn.py --model llama3 \
    --train samples/llama3/samples_binary_all_train.npz \
    --test  samples/llama3/samples_binary_all_test.npz

Usage (legacy: single npz, sample-level split):
  python classifier_cnn.py --model llama3 \
    --samples samples/llama3/samples_binary_all.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==================== Paths ====================
BASE_DIR   = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "samples"

# ==================== Config ====================
RANDOM_SEED = 42


# ==================== Sample loading ====================

def load_samples(path: Path):
    """Load pre-extracted samples from .npz file (output of prepare_samples.py)."""
    data = np.load(path, allow_pickle=False)
    X    = data["X"]   # (N, n_layers, hidden_dim)
    y    = data["y"]   # (N,) int8
    roles = list(data["roles"])
    print(f"  Loaded samples: shape={X.shape}, roles={roles}")
    print(f"  y — steer(1): {(y==1).sum()}, no_steer(0): {(y==0).sum()}")
    return X.astype(np.float32), y.astype(np.int64)


# ==================== Layer range ====================

def parse_layer_range(spec: str, n_layers: int):
    """Parse layer spec: 'all' or 'start-end' (inclusive)."""
    if spec == "all":
        return list(range(n_layers))
    lo, hi = spec.split("-")
    lo, hi = int(lo), int(hi)
    hi = min(hi, n_layers - 1)
    return list(range(lo, hi + 1))


# ==================== Scaler ====================

def scale_per_layer(X: np.ndarray):
    """
    Per-layer StandardScaler. Fits on X in-place.
    X: (N, L, D) — scale each layer independently.
    Returns (scaled X as float32, list of fitted scalers).
    """
    _, L, _ = X.shape
    X = X.astype(np.float32)
    scalers = []
    for l in range(L):
        sc = StandardScaler()
        X[:, l, :] = sc.fit_transform(X[:, l, :])
        scalers.append(sc)
    return X, scalers


# ==================== Model ====================

class LayerAttentionCNN(nn.Module):
    """
    1D-CNN over layers + learned layer attention → binary classifier.

    Input : (B, L, D)   L=num_layers, D=hidden_dim (after projection)
    Output: (B, 2)      logits for classes 0, 1
    """
    def __init__(self, num_layers: int, hidden_dim: int,
                 proj_dim: int = 128, cnn_channels: int = 256,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()

        # Project each layer's hidden dim → proj_dim (weight shared across layers)
        self.proj = nn.Linear(hidden_dim, proj_dim)

        # 1D-CNN: treats layer axis as sequence
        # Input to conv: (B, proj_dim, L)  [channels first]
        self.cnn = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.GELU(),
        )

        # Layer attention: scalar score per layer → softmax weights
        self.attn = nn.Linear(cnn_channels, 1)

        # MLP head → 2 classes
        self.head = nn.Sequential(
            nn.Linear(cnn_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: (B, L, D)
        x = self.proj(x)                    # (B, L, proj_dim)
        x = x.permute(0, 2, 1)             # (B, proj_dim, L)
        x = self.cnn(x)                     # (B, cnn_channels, L)
        x = x.permute(0, 2, 1)             # (B, L, cnn_channels)

        # Attention over layers
        scores  = self.attn(x).squeeze(-1)              # (B, L)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        x = (x * weights).sum(dim=1)                   # (B, cnn_channels)

        return self.head(x)                             # (B, 2)


# ==================== Training ====================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds  = logits.argmax(1)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        correct += (preds == y_batch).sum().item()
        n       += len(y_batch)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels, all_probs


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--train",   default=None,
                        help="Path to train .npz (question-level split)")
    parser.add_argument("--test",    default=None,
                        help="Path to test .npz (question-level split)")
    parser.add_argument("--samples", type=str, default=None,
                        help="Legacy: single .npz with sample-level split fallback")
    parser.add_argument("--layers",  default="all",
                        help="Layer range: 'all' or 'start-end', e.g. '10-25'")
    parser.add_argument("--proj_dim",     type=int,   default=64)
    parser.add_argument("--cnn_channels", type=int,   default=64)
    parser.add_argument("--kernel_size",  type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    # ── Resolve paths ──
    train_path = Path(args.train) if args.train else None
    test_path  = Path(args.test)  if args.test  else None
    use_split  = train_path and train_path.exists()

    if not use_split:
        if args.samples:
            samples_path = Path(args.samples)
        else:
            samples_path = SAMPLE_DIR / args.model / "samples_binary_all.npz"
    else:
        samples_path = None

    print(f"\n{'='*55}")
    print(f"  ConfSteer CNN Classifier (Binary)")
    print(f"  Model   : {args.model}")
    if use_split:
        print(f"  Train   : {train_path}")
        print(f"  Test    : {test_path}")
    else:
        print(f"  Samples : {samples_path}")
    print(f"  Layers  : {args.layers}")
    print(f"  Device  : {device}")
    print(f"{'='*55}\n")

    # ── Load samples ──
    if use_split:
        print("[1] Loading train/test samples (question-level split)...")
        X_tr, y_train = load_samples(train_path)
        X_te, y_test  = load_samples(test_path)

        _, n_layers, D = X_tr.shape
        layer_indices = parse_layer_range(args.layers, n_layers)
        X_tr = X_tr[:, layer_indices, :]
        X_te = X_te[:, layer_indices, :]
        L = len(layer_indices)
        print(f"  Using {L} layers: {layer_indices[0]}–{layer_indices[-1]}")
        print(f"  Train shape: {X_tr.shape}  steer(1): {(y_train==1).sum()}, no_steer(0): {(y_train==0).sum()}")
        print(f"  Test  shape: {X_te.shape}  steer(1): {(y_test==1).sum()},  no_steer(0): {(y_test==0).sum()}")

        # Per-layer scaling: fit on train, apply to test
        print("\n[2] Scaling (per-layer StandardScaler, fit on train)...")
        _, scalers = scale_per_layer(X_tr)
        for l, sc in enumerate(scalers):
            X_te[:, l, :] = sc.transform(X_te[:, l, :])
        X_train, X_test = X_tr, X_te

    else:
        print("[1] Loading pre-extracted samples (legacy single npz)...")
        X, y = load_samples(samples_path)

        _, n_layers, D = X.shape
        layer_indices = parse_layer_range(args.layers, n_layers)
        X = X[:, layer_indices, :]
        L = len(layer_indices)
        print(f"  Using {L} layers: {layer_indices[0]}–{layer_indices[-1]}")
        print(f"  Feature shape: {X.shape}")

        print("\n[2] Scaling (per-layer StandardScaler)...")
        X, _ = scale_per_layer(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

    print(f"  Train: {len(y_train)}  Test: {len(y_test)}")
    print(f"  Train steer(1): {(y_train==1).sum()}  no_steer(0): {(y_train==0).sum()}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ── Build model ──
    model = LayerAttentionCNN(
        num_layers=L,
        hidden_dim=D,
        proj_dim=args.proj_dim,
        cnn_channels=args.cnn_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[3] Model: {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──
    print(f"\n[4] Training ({args.epochs} epochs, batch={args.batch}, lr={args.lr})...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'-'*50}")

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        marker = " ←" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  {val_loss:>8.4f}  {val_acc:>7.3f}{marker}")

    # ── Final evaluation with best weights ──
    model.load_state_dict(best_state)
    _, _, all_preds, all_labels, all_probs = evaluate(model, test_loader, criterion, device)

    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n[5] Evaluation on test set (best checkpoint):")
    print(classification_report(all_labels, all_preds,
                                 target_names=["no_steer(0)", "steer(+1)"]))
    print(f"  ROC-AUC: {auc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("  labels: 0, +1")
    print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))

    # ── Save model ──
    layer_tag = args.layers.replace("-", "_")
    out_path  = BASE_DIR / f"cnn_{args.model}_{layer_tag}.pt"
    torch.save({
        "model_state":   best_state,
        "layer_indices": layer_indices,
        "num_layers":    L,
        "hidden_dim":    D,
        "proj_dim":      args.proj_dim,
        "cnn_channels":  args.cnn_channels,
        "kernel_size":   args.kernel_size,
    }, out_path)
    print(f"\nModel saved: {out_path}")


if __name__ == "__main__":
    main()

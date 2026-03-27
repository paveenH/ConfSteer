"""
classifier_transformer.py — Per-layer PCA + Transformer Encoder
================================================================
Architecture:
  Input (N, L, D)
    → Per-layer StandardScaler (fit on train)
    → Per-layer PCA (D → pca_dim, fit on train)      [same as PCA-CNN]
    → Transformer Encoder (2 layers, nhead=4)         [captures cross-layer interactions]
    → Mean pooling over layers                        [aggregate layer representations]
    → MLP head → 2 classes

Key difference from PCA-CNN:
  PCA-CNN uses 1D-CNN (local layer patterns, kernel=3).
  Transformer captures arbitrary cross-layer relationships (e.g. layer 5 + layer 20).

Usage:
  python classifier_transformer.py --model llama3 \\
    --train samples/llama3/samples_orig_all_25k_train.npz \\
    --test  samples/llama3/samples_orig_all_25k_test.npz \\
    --pca_dim 128
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
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


def fit_scale_pca(X_tr: np.ndarray, X_te: np.ndarray, pca_dim: int):
    """Per-layer StandardScaler + PCA. Fit on train, transform both."""
    N_tr, L, D = X_tr.shape
    N_te       = X_te.shape[0]
    out_tr = np.zeros((N_tr, L, pca_dim), dtype=np.float32)
    out_te = np.zeros((N_te, L, pca_dim), dtype=np.float32)

    for l in range(L):
        sc = StandardScaler()
        X_tr_l = sc.fit_transform(X_tr[:, l, :])
        X_te_l = sc.transform(X_te[:, l, :])

        n_comp = min(pca_dim, D, N_tr)
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        out_tr[:, l, :n_comp] = pca.fit_transform(X_tr_l)
        out_te[:, l, :n_comp] = pca.transform(X_te_l)

    return out_tr, out_te


# ==================== Model ====================

class TransformerClassifier(nn.Module):
    """
    Transformer Encoder over layers.

    Input : (B, L, pca_dim)
    Steps :
      1. Linear input projection: pca_dim → d_model
      2. Learnable positional embedding (one per layer)
      3. Transformer Encoder (num_layers=2, nhead=4)
      4. Mean pooling over L → (B, d_model)
      5. MLP head → 2 classes
    """
    def __init__(self, num_layers: int, pca_dim: int,
                 d_model: int = 128, nhead: int = 4,
                 num_encoder_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.3):
        super().__init__()

        self.input_proj = nn.Linear(pca_dim, d_model)
        self.pos_emb    = nn.Embedding(num_layers, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, L, d_model)
            norm_first=True,    # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B, L, pca_dim)
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device)          # (L,)
        x = self.input_proj(x) + self.pos_emb(pos)      # (B, L, d_model)
        x = self.encoder(x)                              # (B, L, d_model)
        x = x.mean(dim=1)                                # (B, d_model)  mean pooling
        return self.head(x)                              # (B, 2)


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
    parser.add_argument("--pca_dim",  type=int,   default=128)
    parser.add_argument("--d_model",  type=int,   default=128,
                        help="Transformer hidden size")
    parser.add_argument("--nhead",    type=int,   default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward",    type=int, default=256)
    parser.add_argument("--dropout",  type=float, default=0.3)
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--device",   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  PCA + Transformer Encoder Classifier")
    print(f"  Model      : {args.model}")
    print(f"  Train      : {args.train}")
    print(f"  Test       : {args.test}")
    print(f"  PCA dim    : {args.pca_dim}")
    print(f"  d_model    : {args.d_model}  nhead={args.nhead}  enc_layers={args.num_encoder_layers}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    print("[1] Loading samples...")
    X_tr, y_train = load_samples(Path(args.train))
    X_te, y_test  = load_samples(Path(args.test))
    _, L, D = X_tr.shape

    print(f"\n[2] Per-layer StandardScaler + PCA({args.pca_dim})...")
    X_train, X_test = fit_scale_pca(X_tr, X_te, args.pca_dim)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = TransformerClassifier(
        num_layers=L, pca_dim=args.pca_dim,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
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

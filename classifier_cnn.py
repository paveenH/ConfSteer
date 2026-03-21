"""
ConfSteer CNN Classifier
========================
Uses ALL layers of hidden states (N, num_layers, hidden_dim) to predict
whether +4 steering (+1), -4 steering (-1), or no steering (0) is optimal.

Architecture:
  Input (N, L, D)
    → Linear projection: D → proj_dim (per layer, shared weights)
    → 1D-CNN over layers (kernel_size=3)
    → Layer Attention: learned weighted sum over layers
    → MLP head → 3 classes

Label definition (same as classifier_demo.py):
  +1  : label_pos4 == +1  (wrong→correct under +4)
  -1  : label_neg4 == +1  (wrong→correct under -4)
   0  : all others — downsampled to ratio × max(|+1|, |-1|)

Usage:
  python classifier_cnn.py --model llama3
  python classifier_cnn.py --model qwen3 --epochs 30 --lr 3e-4
  python classifier_cnn.py --model llama3 --layers 10-25   # subset of layers
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== Paths ====================
BASE_DIR   = Path(__file__).parent
LABEL_DIR  = BASE_DIR / "labels"
HIDDEN_DIR = BASE_DIR / "HiddenStates"

# ==================== Config ====================
RANDOM_SEED     = 42
NO_CHANGE_RATIO = 1.0   # class-0 samples = ratio × max(|+1|, |-1|)


# ==================== H5 path mapping (same as classifier_demo) ====================

def get_h5_path(model: str, task: str, orig_stem: str) -> Path:
    stem = orig_stem
    for suffix in ["_answers", "_mc1", "_mc2"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    h5_task = "truthfulqa" if task.startswith("tqa") else task
    h5_name = f"neutral_{stem}.h5"
    return HIDDEN_DIR / model / h5_task / h5_name


# ==================== Label loading ====================

def assign_class(lp: int, ln: int) -> int:
    if lp == 1:
        return 1
    if ln == 1:
        return -1
    return 0


def load_labels(model: str):
    result = []
    pattern = str(LABEL_DIR / model / "**" / "labels_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        p = Path(path)
        task = p.parent.name
        orig_stem = p.stem[len("labels_"):]
        with open(p) as f:
            data = json.load(f)["data"]
        result.append((p, task, orig_stem, data))
    return result


# ==================== Feature extraction ====================

def parse_layer_range(spec: str, n_layers: int):
    """Parse layer spec: 'all' or 'start-end' (inclusive)."""
    if spec == "all":
        return list(range(n_layers))
    lo, hi = spec.split("-")
    lo, hi = int(lo), int(hi)
    hi = min(hi, n_layers - 1)
    return list(range(lo, hi + 1))


def extract_features(model: str, label_files, layer_spec: str = "all"):
    """
    Load hidden states and return X of shape (N, selected_layers, hidden_dim).
    First file determines n_layers; layer_spec is resolved then.
    """
    import h5py

    X, y = [], []
    skipped = 0
    layer_indices = None   # resolved on first file

    for label_path, task, orig_stem, samples in label_files:
        h5_path = get_h5_path(model, task, orig_stem)
        if not h5_path.exists():
            print(f"  [skip] h5 not found: {h5_path}")
            skipped += 1
            continue

        with h5py.File(h5_path, "r") as hf:
            hs = hf["hidden_states"]          # (N, num_layers, hidden_dim)
            n_layers = hs.shape[1]

            if layer_indices is None:
                layer_indices = parse_layer_range(layer_spec, n_layers)
                print(f"  Using {len(layer_indices)} layers: {layer_indices[0]}–{layer_indices[-1]}")

            for s in samples:
                cls = assign_class(s["label_pos4"], s["label_neg4"])
                idx = s["index"]
                feat = hs[idx, layer_indices, :]   # (selected_layers, hidden_dim)
                X.append(feat)
                y.append(cls)

    print(f"  Loaded {len(X)} samples ({skipped} files skipped)")
    return np.array(X, dtype=np.float32), np.array(y)


# ==================== Sampling ====================

def balanced_sample(X, y, ratio=NO_CHANGE_RATIO, seed=RANDOM_SEED):
    rng = random.Random(seed)
    idx_pos  = np.where(y ==  1)[0].tolist()
    idx_neg  = np.where(y == -1)[0].tolist()
    idx_zero = np.where(y ==  0)[0].tolist()

    n_keep = int(ratio * max(len(idx_pos), len(idx_neg)))
    idx_zero_sampled = rng.sample(idx_zero, min(n_keep, len(idx_zero)))

    keep = idx_pos + idx_neg + idx_zero_sampled
    rng.shuffle(keep)
    keep = np.array(keep)
    return X[keep], y[keep]


# ==================== Model ====================

class LayerAttentionCNN(nn.Module):
    """
    1D-CNN over layers + learned layer attention → MLP classifier.

    Input : (B, L, D)   L=num_layers, D=hidden_dim
    Output: (B, 3)      logits for classes -1, 0, +1
    """
    def __init__(self, num_layers: int, hidden_dim: int,
                 proj_dim: int = 128, cnn_channels: int = 256,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()

        # Project each layer's hidden dim to proj_dim (weight shared across layers)
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

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(cnn_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        # x: (B, L, D)
        x = self.proj(x)                    # (B, L, proj_dim)
        x = x.permute(0, 2, 1)             # (B, proj_dim, L)
        x = self.cnn(x)                     # (B, cnn_channels, L)
        x = x.permute(0, 2, 1)             # (B, L, cnn_channels)

        # Attention over layers
        scores = self.attn(x).squeeze(-1)   # (B, L)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        x = (x * weights).sum(dim=1)        # (B, cnn_channels)

        return self.head(x)                 # (B, 3)


# ==================== Training ====================

CLASS_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_CLASS = {0: -1, 1: 0, 2: 1}


def to_idx(y):
    return np.array([CLASS_TO_IDX[c] for c in y])


def make_weighted_sampler(y_idx):
    """WeightedRandomSampler for balanced mini-batches."""
    counts = np.bincount(y_idx, minlength=3).astype(float)
    weights_per_class = 1.0 / np.where(counts == 0, 1, counts)
    sample_weights = torch.tensor([weights_per_class[i] for i in y_idx], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        n += len(y_batch)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="llama3", choices=["llama3", "qwen3"])
    parser.add_argument("--layers",  default="all",
                        help="Layer range: 'all' or 'start-end', e.g. '10-25'")
    parser.add_argument("--proj_dim",     type=int, default=128)
    parser.add_argument("--cnn_channels", type=int, default=256)
    parser.add_argument("--kernel_size",  type=int, default=3)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device(args.device)

    print(f"\n{'='*55}")
    print(f"  ConfSteer CNN Classifier")
    print(f"  Model  : {args.model}")
    print(f"  Layers : {args.layers}")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    # ── Load labels ──
    print("[1] Loading labels...")
    label_files = load_labels(args.model)
    print(f"  Found {len(label_files)} label files")

    # ── Extract all-layer features ──
    print("\n[2] Extracting features (all layers)...")
    X, y = extract_features(args.model, label_files, args.layers)
    # X: (N, L, D)
    print(f"  Feature shape: {X.shape}")
    print(f"  Raw class dist — +1: {(y==1).sum()}, -1: {(y==-1).sum()}, 0: {(y==0).sum()}")

    # ── Downsample class 0 ──
    print("\n[3] Balancing class 0...")
    X, y = balanced_sample(X, y)
    print(f"  Balanced — +1: {(y==1).sum()}, -1: {(y==-1).sum()}, 0: {(y==0).sum()}")
    print(f"  Total samples: {len(y)}")

    # ── Standardize per-feature across samples ──
    # Reshape to (N, L*D), scale, reshape back
    N, L, D = X.shape
    X_flat = X.reshape(N, L * D)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat).astype(np.float32)
    X = X_flat.reshape(N, L, D)

    # ── Train/test split ──
    y_idx = to_idx(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y_idx
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    sampler     = make_weighted_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ── Build model ──
    num_layers = L
    hidden_dim = D
    model = LayerAttentionCNN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        proj_dim=args.proj_dim,
        cnn_channels=args.cnn_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[4] Model: {n_params:,} trainable parameters")

    # Class weights for loss (extra boost for neg/-1 which is fewest)
    counts = np.bincount(y_train, minlength=3).astype(float)
    class_weights = torch.tensor(1.0 / np.where(counts == 0, 1, counts),
                                  dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──
    print(f"\n[5] Training ({args.epochs} epochs, batch={args.batch}, lr={args.lr})...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'-'*50}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        marker = " ←" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  {val_loss:>8.4f}  {val_acc:>7.3f}{marker}")

    # ── Final evaluation with best weights ──
    model.load_state_dict(best_state)
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    all_preds  = [IDX_TO_CLASS[p] for p in all_preds]
    all_labels = [IDX_TO_CLASS[l] for l in all_labels]

    print(f"\n[6] Evaluation on test set (best checkpoint):")
    print(classification_report(all_labels, all_preds,
                                 target_names=["neg(-1)", "no_change(0)", "pos(+1)"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print("  labels: -1, 0, +1")
    print(confusion_matrix(all_labels, all_preds, labels=[-1, 0, 1]))

    # ── Save model ──
    out_path = BASE_DIR / f"cnn_{args.model}_{args.layers.replace('-','_')}.pt"
    torch.save({"model_state": best_state, "scaler": scaler,
                "layer_indices": args.layers, "num_layers": L, "hidden_dim": D,
                "proj_dim": args.proj_dim, "cnn_channels": args.cnn_channels,
                "kernel_size": args.kernel_size}, out_path)
    print(f"\nModel saved: {out_path}")


if __name__ == "__main__":
    main()

"""
Absolute-pitch experiments: octave-invariant pitch-class (chroma) representation.

Motivation (see presentation "Toward an Absolute Pitch Model"):
  Instead of a flat 88-way classifier, decompose the pitch space into
  * register  = note index // 12  (0..7, i.e. A0..G#0 ... C8)
  * chroma    = note index % 12   (the pitch class, independent of octave)
  and learn *compressed* representations that exploit octave equivalence,
  as humans acquiring absolute pitch do (learning within a limited range
  and generalising via harmonic/octave structure).

  note_index = register * 12 + chroma
  -> 12 chroma + 8 register units encode all 88 keys (88 -> 20 units).

Experiments implemented:
  1. chroma_full       Train a 12-unit chroma head on ALL 88 notes. Chance = 8.3%.
  2. chroma_transfer   Train the chroma head on ONE octave (C4..B4, register 3)
                       and measure octave-equivalence transfer to every register.
  3. chroma_register   Train separate chroma (12) and register (8) heads, then
                       reconstruct the precise note and report 88-way accuracy.
  4. probe             Record hidden-layer spike activity and quantify how well
                       the representation is organised by chroma (t-SNE 2-D).

Run (examples):
  uv run python experiments/absolute_pitch.py --version handy --exp chroma_full --epochs 20
  uv run python experiments/absolute_pitch.py --version handy --exp chroma_transfer --epochs 20
  uv run python experiments/absolute_pitch.py --version handy --exp chroma_register --epochs 20 --epochs2 20
  uv run python experiments/absolute_pitch.py --version handy --exp probe --limit 400
  # quick sanity check:
  uv run python experiments/absolute_pitch.py --smoke
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.snn import PitchSNN

try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = ROOT / "data" / "plots"


# ---------------------------------------------------------------------------
# Label decomposition: register / chroma from the 88-key index
# ---------------------------------------------------------------------------
def chrono_labels(note_idx):
    """note_idx (N,) in [0, 87]; returns (register, chroma) each (N,)."""
    note_idx = note_idx % 88
    return note_idx // 12, note_idx % 12


# ---------------------------------------------------------------------------
# Model with a probe hook on hidden LIF activity (drops-in for PitchSNN)
# ---------------------------------------------------------------------------
class ProbePitchSNN(PitchSNN):
    def forward(self, x, return_hidden=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem2_rec = []
        spk1_rec = []
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
        mem2_rec = torch.stack(mem2_rec)
        if return_hidden:
            return mem2_rec, torch.stack(spk1_rec)
        return mem2_rec


# ---------------------------------------------------------------------------
# I/O and helper plumbing
# ---------------------------------------------------------------------------
def load_dataset(version):
    path = ROOT / "data" / f"piano_dataset_{version}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run src/generate_piano_dataset.py first.")
    data = torch.load(path, map_location="cpu")
    X, y = data["X"], data["y"]  # (8800, 100, 50) if 2 versions x 50/class else (N,100,50)
    return X, y


def make_loaders(X, y, train_idx, test_idx, batch_size):
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    test_ds = TensorDataset(X[test_idx], y[test_idx])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_model(model, train_loader, test_loader, epochs, lr, device, verbose=True):
    """Supervised surrogate-gradient training on time-averaged membrane logits."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b = X_b.transpose(0, 1).to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b).mean(dim=0)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y_b).sum().item()
            seen += len(y_b)
        acc = 100.0 * correct / seen
        hist.append(acc)
        if verbose:
            print(f"  epoch {epoch+1:02d}/{epochs}  loss={total_loss:.4f}  acc={acc:.2f}%")
    return test_accuracy(model, test_loader, device)


def test_accuracy(model, loader, device):
    model.eval()
    correct, seen = 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.transpose(0, 1).to(device)
            y_b = y_b.to(device)
            logits = model(X_b).mean(dim=0)
            correct += (logits.argmax(dim=1) == y_b).sum().item()
            seen += len(y_b)
    return 100.0 * correct / seen


def per_register_accuracy(model, X, labels, note_idx, device):
    """Report label accuracy separately for each piano register.

    note_idx: array of 88-key note indices (used to derive the register);
    labels:   the classification labels evaluated by the model.
    """
    model.eval()
    out = {}
    with torch.no_grad():
        for reg_i in sorted(set((note_idx // 12).tolist())):
            mask = (note_idx // 12) == reg_i
            sub = note_idx[mask.nonzero(as_tuple=True)[0]]
            if len(sub) == 0:
                continue
            acc = test_accuracy(model, DataLoader(TensorDataset(X[sub], labels[sub]), batch_size=64, shuffle=False), device)
            out[int(reg_i)] = acc
    return out


# ---------------------------------------------------------------------------
# Experiment 1: chroma trained on the full 88-key range
# ---------------------------------------------------------------------------
def exp_chroma_full(version, epochs, lr, batch_size, limit, device):
    X, y = load_dataset(version)
    idx = torch.randperm(len(y))[:limit] if limit else torch.arange(len(y))
    y_chroma = y % 12
    n = len(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    train_loader, test_loader = make_loaders(X, y_chroma, train_idx, test_idx, batch_size)
    model = PitchSNN(num_outputs=12).to(device)
    test_acc = train_model(model, train_loader, test_loader, epochs, lr, device)
    print(f"[chroma_full] 12 chroma units, all 88 notes -> test chroma acc = {test_acc:.2f}% "
          f"(chance 8.3%%)")


# ---------------------------------------------------------------------------
# Experiment 2: single-octave training + octave-equivalence transfer
# ---------------------------------------------------------------------------
def exp_chroma_transfer(version, epochs, lr, batch_size, limit, device, train_register=3):
    X, y = load_dataset(version)
    reg, chroma = chrono_labels(y)
    oct_idx = torch.nonzero(reg == train_register).flatten()      # e.g. C4..B4
    if limit:
        perm = torch.randperm(len(oct_idx))[: limit]
        oct_idx = oct_idx[perm]
    n = len(oct_idx)
    split = int(0.8 * n)
    train_idx, test_idx = oct_idx[:split], oct_idx[split:]

    train_loader, test_loader = make_loaders(X, chroma, train_idx, test_idx, batch_size)
    model = PitchSNN(num_outputs=12).to(device)
    test_acc = train_model(model, train_loader, test_loader, epochs, lr, device)
    print(f"[chroma_transfer] trained on register {train_register} -> held-out chroma acc "
          f"(same register) = {test_acc:.2f}%")

    # Octave-equivalence probe: all notes, per register
    print("  octave-equivalence transfer (chroma accuracy per register):")
    for reg_i, acc in per_register_accuracy(model, X, chroma, y, device).items():
        print(f"    register {reg_i} (notes A{reg_i}..G#{reg_i}): {acc:.2f}%")
    return model


# ---------------------------------------------------------------------------
# Experiment 3: chroma + register heads -> precise note reconstruction
# ---------------------------------------------------------------------------
def exp_chroma_register(version, epochs_chroma, epochs_reg, lr, batch_size, limit, device):
    X, y = load_dataset(version)
    reg, chroma = chrono_labels(y)
    idx = torch.randperm(len(y))[:limit] if limit else torch.arange(len(y))
    n = len(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    print("[chroma_register] training chroma head (12 units)...")
    chroma_model = PitchSNN(num_outputs=12).to(device)
    tr, te = make_loaders(X, chroma, train_idx, test_idx, batch_size)
    chroma_acc = train_model(chroma_model, tr, te, epochs_chroma, lr, device)

    print("[chroma_register] training register head (8 units)...")
    reg_model = PitchSNN(num_outputs=8).to(device)
    tr, te = make_loaders(X, reg, train_idx, test_idx, batch_size)
    reg_acc = train_model(reg_model, tr, te, epochs_reg, lr, device)

    # Combined reconstruction on the test split
    chroma_model.eval(); reg_model.eval()
    correct, seen = 0, 0
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            sub = test_idx[i:i + batch_size]
            Xb = X[sub].transpose(0, 1).to(device)
            c_pred = chroma_model(Xb).mean(dim=0).argmax(dim=1)
            r_pred = reg_model(Xb).mean(dim=0).argmax(dim=1)
            note_pred = r_pred * 12 + c_pred
            correct += (note_pred.cpu() == y[sub]).sum().item()
            seen += len(sub)
    note_acc = 100.0 * correct / seen
    print(f"[chroma_register] chroma={chroma_acc:.2f}%  register={reg_acc:.2f}%  "
          f"reconstructed 88-note acc = {note_acc:.2f}% (chance 1.1%%)")


# ---------------------------------------------------------------------------
# Experiment 4: representation probe (hidden LIF activity, t-SNE by chroma)
# ---------------------------------------------------------------------------
def exp_probe(version, epochs, lr, batch_size, limit, device, save_fig=True):
    X, y = load_dataset(version)
    X = X[:limit]; y = y[:limit]
    train_loader, test_loader = make_loaders(X, y % 12, torch.arange(len(y) - len(y) // 4),
                                             torch.arange(len(y) - len(y) // 4, len(y)), batch_size)
    model = ProbePitchSNN(num_outputs=12).to(device)
    train_model(model, train_loader, test_loader, epochs, lr, device)

    model.eval()
    hidden = []
    labels = []
    with torch.no_grad():
        for X_b, y_b in DataLoader(TensorDataset(X, y % 12), batch_size=64, shuffle=False):
            _, spk1 = model(X_b.transpose(0, 1).to(device), return_hidden=True)
            hidden.append(spk1.sum(dim=0).cpu())       # (B, 100) hidden spike counts
            labels.append(y_b)
    H = torch.cat(hidden).numpy()
    L = torch.cat(labels).numpy()

    # Chroma purity of the hidden population code (skip classes with no samples)
    present = [c for c in range(12) if (L == c).any()]
    if len(present) >= 2:
        centroids = np.stack([H[L == c].mean(axis=0) for c in present])
        nn_assignment = np.array(present)[(H @ centroids.T).argmax(axis=1)]
        purity = 100.0 * (nn_assignment == L).mean()
        print(f"[probe] hidden-spike nearest-centroid chroma purity = {purity:.2f}% "
              f"(chance 8.3%%)")
    else:
        purity = float("nan")
        print("[probe] too few classes in this run to compute chroma purity")

    if save_fig and _HAS_SKLEARN:
        proj = TSNE(n_components=2, init="pca", random_state=0).fit_transform(H)
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=L, cmap="tab20", s=8, alpha=0.8)
        fig.colorbar(sc, ax=ax, ticks=range(12), label="pitch class (chroma)")
        ax.set_title(f"Hidden LIF population code (t-SNE) | chroma purity {purity:.1f}%")
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PLOT_DIR / f"tsne_chroma_probe_{version}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[probe] t-SNE saved to {out_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Absolute-pitch (chroma/register) experiments")
    p.add_argument("--version", default="handy", choices=["handy", "spikify"])
    p.add_argument("--exp", default="chroma_full",
                   choices=["chroma_full", "chroma_transfer", "chroma_register", "probe"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--epochs2", type=int, default=20, help="register-head epochs (chroma_register)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of samples (for quick smoke runs)")
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to verify the pipeline end-to-end")
    a = p.parse_args()

    device = torch.device(a.device)
    print(f"device: {device} | version: {a.version} | experiment: {a.exp}")
    if a.smoke:
        a = argparse.Namespace(**{**vars(a), "epochs": 1, "epochs2": 1, "limit": 60})

    try:
        if a.exp == "chroma_full":
            exp_chroma_full(a.version, a.epochs, a.lr, a.batch_size, a.limit, device)
        elif a.exp == "chroma_transfer":
            exp_chroma_transfer(a.version, a.epochs, a.lr, a.batch_size, a.limit, device)
        elif a.exp == "chroma_register":
            exp_chroma_register(a.version, a.epochs, a.epochs2, a.lr, a.batch_size, a.limit, device)
        elif a.exp == "probe":
            exp_probe(a.version, a.epochs, a.lr, a.batch_size, a.limit, device)
    except FileNotFoundError as e:
        print(f"Missing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
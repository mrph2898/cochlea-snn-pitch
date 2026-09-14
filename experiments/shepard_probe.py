"""
Shepard/tritone battery (S1) for the absolute-pitch pipeline.

What it tests, and the biology behind each condition:

A. Chroma transfer to octave-ambiguous spectra.
   A Shepard tone (Shepard 1964) is built entirely from octave-spaced
   partials, so its spectral pattern is invariant under octave translation:
   pitch *class* is present, pitch *height* is absent. If the chroma head
   trained on piano notes generalises to such tones, the model really solved
   octave equivalence in the cochlear place map (true "chroma" in line with
   the Warren et al. 2003 anterior-auditory chroma representation), rather
   than latching onto piano-specific harmonic-envelope cues.

B. Place vs temporal coding of chroma.
   phase == "cos": cosine-phase partials -> sharp waveform peaks -> strong
   volley/periodicity (temporal) cues (Rose et al. 1967; Cariani & Delgutte
   1996; Saddler et al. 2021 - auditory-nerve phase locking is needed for
   human-like pitch).
   phase == "rand": random phases -> no temporal structure, only the
   tonotopic (place/rate) profile (Oxenham 2012).
   Comparing chroma accuracy between the two conditions attributes the
   model's pitch-class readout to the cochlea's place code, temporal code,
   or both.

C. Height ambiguity: normalises toward chance without breaking chroma.
   Pitch height should PLACE-CODE the spectral envelope even when temporal
   height is ambiguous -> register argmax should track the spectral centroid
   (place/rate cue) monotonically - or collapse to uniform if the head relied
   on temporal/absolute fundamentals only.

D. Tritone paradox (Deutsch 1986).
   Pairs a half-octave apart are directionally ambiguous in perception; the
   perceived up/down direction depends on pitch-class statistics of the
   listener's "musical diet". We measure the model's *internal consistency*:
   rate(up) + rate(down) over trials approximating 1 for a deterministic
   directional readout, and report the per-pitch-class ascending-rate pattern
   (a model trained only on in-key piano statistics may show a Shepard/Diamond
   style bias, or none at all).

Chance levels: chroma 1/12 = 8.33%; register 1/8 = 12.5%.

Run:
  uv run python experiments/shepard_probe.py --version both --exp battery
  uv run python experiments/shepard_probe.py --version handy --exp chroma --n-eval 4
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from absolute_pitch import (  # noqa: E402
    load_dataset, make_loaders, train_model, chrono_labels,
    PLOT_DIR, DEVICE,
)
from src.shepard import shepard_tone, spectral_centroid_hz  # noqa: E402
from src.preprocess import (  # noqa: E402
    add_white_noise, gammatone_filterbank, ihc_rectification_compression,
    generate_poisson_spikes,
)
from spikify.filters import FilterBank  # noqa: E402
from spikify.encoders.rate import poisson  # noqa: E402

FS = 44100
DURATION = 0.1
NUM_CHANNELS = 50
BIN_SIZE = int(FS / 1000)  # 1 ms bins

CHANCE_CHROMA = 100.0 / 12
CHANCE_REG = 100.0 / 8


# --------------------------------------------------------------------------
# Front-end: identical encoding to src/generate_piano_dataset.py
# --------------------------------------------------------------------------
FILTERBANKS = {}


def get_filterbank(version):
    """Cache the spikify FilterBank exactly as built for the piano dataset."""
    if version not in FILTERBANKS:
        FILTERBANKS[version] = FilterBank(
            fs=FS, channels=NUM_CHANNELS, f_min=27.5, f_max=8000.0,
            filter_type="gammatone", order=4)
    return FILTERBANKS[version]


def encode_waveform(wave, version, snr_db=None):
    """Bin a waveform into the (time=100, channels=50) spike frame.

    Mirrors generate_piano_dataset.generate_spike_tensor:
      handy   : custom gammatone (100–6500 Hz) + IHC + Poisson (max 800 Hz)
      spikify : FilterBank (27.5–8000 Hz) gammatone + spikify Poisson
    """
    if snr_db is not None:
        wave = add_white_noise(wave, snr_db)

    if version == "spikify":
        filtered = get_filterbank(version).decompose(wave)   # (T, C, 1)
        filtered = filtered.reshape(-1, filtered.shape[1] * filtered.shape[2])
        spikes = poisson(filtered, interval_length=2).T      # (C, T)
    else:  # handy
        filtered, _ = gammatone_filterbank(wave, FS, num_channels=NUM_CHANNELS)
        ihc = ihc_rectification_compression(filtered)
        spikes = generate_poisson_spikes(ihc, FS, max_rate=800)  # (C, T)

    num_bins = spikes.shape[1] // BIN_SIZE
    binned = np.zeros((NUM_CHANNELS, num_bins))
    for i in range(num_bins):
        window = spikes[:, i * BIN_SIZE:(i + 1) * BIN_SIZE]
        binned[:, i] = (window.sum(axis=1) > 0).astype(float)
    return torch.tensor(binned, dtype=torch.float32).T         # (time, ch)


# --------------------------------------------------------------------------
# Training heads (same recipe as absolute_pitch)
# --------------------------------------------------------------------------
def train_head(X, y, num_classes, epochs, lr, batch_size, device, limit=None):
    idx = torch.randperm(len(y))[:limit] if limit else torch.randperm(len(y))
    n = len(idx)
    split = int(0.8 * n)
    tr, te = make_loaders(X, y, idx[:split], idx[split:], batch_size)
    from src.snn import PitchSNN
    model = PitchSNN(num_outputs=num_classes).to(device)
    acc = train_model(model, tr, te, epochs, lr, device, verbose=False)
    return model, acc


def predict(model, X, device, batch_size=256):
    """Time-averaged logit argmax for a (N, T, C) tensor."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].transpose(0, 1).to(device)
            logits = model(xb).mean(dim=0)
            outs.append(logits.argmax(dim=1).cpu())
    return torch.cat(outs)


# --------------------------------------------------------------------------
# Shepard stimulus sets
# --------------------------------------------------------------------------
def build_shepard_set(version, chromas, env_centers, phases, n_eval,
                      snr_bounds=(10.0, 30.0), seed=0):
    """Encode single Shepard tones, one per (chroma, env, phase, trial)."""
    X_list, chroma_l, env_l, phase_l, cent_l = [], [], [], [], []
    counter = 0
    for c in chromas:
        for env in env_centers:
            for ph in phases:
                for _ in range(n_eval):
                    np.random.seed(seed + 10_000_000 * 0 + counter)
                    t, sig, freqs, env_w = shepard_tone(
                        c, env_center_hz=env, duration=DURATION, fs=FS,
                        phase=ph)
                    snr = float(np.random.uniform(*snr_bounds))
                    X_list.append(encode_waveform(sig, version, snr_db=snr))
                    chroma_l.append(c)
                    env_l.append(env)
                    phase_l.append(ph)
                    cent_l.append(spectral_centroid_hz(freqs, env_w))
                    counter += 1
    X = torch.stack(X_list)
    return X, np.array(chroma_l), np.array(env_l), np.array(phase_l), np.array(cent_l)


def build_tritone_set(version, n_eval, env=800.0, phases=("cos", "rand"),
                      snr_bounds=(10.0, 30.0), seed=0):
    """Encode tritone pairs (c, c+6) in both orders, per phase."""
    pairs = []  # list of (X_a, X_b, start_chroma, order, phase)
    counter = 0
    for c in range(12):
        c2 = (c + 6) % 12
        for ph in phases:
            for _ in range(n_eval):
                np.random.seed(seed + 10_000_000 * counter)
                t, sa, fa, ea = shepard_tone(c, env_center_hz=env, phase=ph)
                t, sb, fb, eb = shepard_tone(c2, env_center_hz=env, phase=ph)
                snr = float(np.random.uniform(*snr_bounds))
                xa = encode_waveform(sa, version, snr_db=snr)
                snr = float(np.random.uniform(*snr_bounds))
                xb = encode_waveform(sb, version, snr_db=snr)
                pairs.append((xa, xb, c, +1, ph))   # up:  c -> c+6
                pairs.append((xb, xa, c, -1, ph))   # down: (c+6) -> c
                counter += 1
    return pairs


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def exp_battery(version, epochs, epochs2, lr, batch_size, device,
                n_eval, env_centers, phases, tritone_env, seed):
    """Run chroma-transfer, register/height, and tritone analyses."""
    X, y = load_dataset(version)
    torch.manual_seed(seed)

    print(f"\n[{version}] training chroma head (12 units, all 88 notes)...")
    chroma_model, chroma_piano = train_head(X, y % 12, 12, epochs, lr, batch_size, device)
    print(f"[{version}] piano chroma test acc = {chroma_piano:.2f}% (chance 8.3%)")

    print(f"[{version}] training register head (8 units)...")
    reg_model, reg_piano = train_head(X, y // 12, 8, epochs2, lr, batch_size, device)
    print(f"[{version}] piano register test acc = {reg_piano:.2f}% (chance 12.5%)")

    res = {
        "piano_chroma_test": float(chroma_piano),
        "piano_register_test": float(reg_piano),
    }

    # --- A/B: chroma transfer + phase discrimination ---
    chromas = list(range(12))
    Xs, cl, envl, phl, centl = build_shepard_set(
        version, chromas, env_centers, phases, n_eval, seed=seed)
    chroma_pred = predict(chroma_model, Xs, device).numpy()

    acc = 100.0 * (chroma_pred == cl).mean()
    per_chroma = {int(c): 100.0 * (chroma_pred[cl == c] == c).mean()
                  for c in chromas}
    per_phase = {ph: 100.0 * (chroma_pred[phl == ph] == cl[phl == ph]).mean()
                 for ph in phases}
    res["shepard_chroma_acc"] = float(acc)
    res["shepard_chroma_per_pitch"] = per_chroma
    res["shepard_chroma_per_phase"] = per_phase
    print(f"[{version}] Shepard chroma acc = {acc:.2f}% (piano {chroma_piano:.2f}%; "
          f"chance 8.3%)")
    for ph in phases:
        print(f"   phase={ph:4s} -> {per_phase[ph]:.2f}%")

    # --- C: height ambiguity / place-coding of register ---
    reg_pred = predict(reg_model, Xs, device).numpy()
    reg_hist = np.bincount(reg_pred, minlength=8).astype(float)
    p = reg_hist / reg_hist.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    uniform_entropy = float(np.log(8))
    res["register_entropy"] = entropy
    res["register_entropy_uniform"] = uniform_entropy
    res["reg_hist"] = reg_hist.tolist()
    res["env_centers"] = [float(e) for e in env_centers]

    # does the register head track the spectral centroid (place cue)?
    mean_reg_per_env = {}
    for env in env_centers:
        m = reg_pred[envl == env].mean()
        mean_reg_per_env[str(env)] = float(m)
    env_vals = np.array([np.log2(e) for e in env_centers])
    reg_means = np.array([mean_reg_per_env[str(e)] for e in env_centers])
    ok = ~np.isnan(reg_means)
    corr = float(np.corrcoef(env_vals[ok], reg_means[ok])[0, 1]) if ok.sum() > 1 else float("nan")
    res["register_envcentroid_corr"] = corr
    res["register_mean_per_env"] = mean_reg_per_env
    print(f"[{version}] register on Shepard: entropy {entropy:.3f} nat "
          f"(uniform {uniform_entropy:.3f})")
    print(f"[{version}] register tracks spectral centroid: corr = {corr:.3f}")

    # --- D: tritone directionality consistency ---
    pairs = build_tritone_set(version, n_eval, env=tritone_env, seed=seed)

    asc_up = np.zeros(12)
    asc_down = np.zeros(12)
    n_up = np.zeros(12)
    n_down = np.zeros(12)
    for xa, xb, c, order, ph in pairs:
        na = predict(chroma_model, xa.unsqueeze(0), device).item() \
            + 12 * predict(reg_model, xa.unsqueeze(0), device).item()
        nb = predict(chroma_model, xb.unsqueeze(0), device).item() \
            + 12 * predict(reg_model, xb.unsqueeze(0), device).item()
        if order == +1:
            asc_up[c] += (nb > na)
            n_up[c] += 1
        else:
            asc_down[c] += (nb > na)
            n_down[c] += 1
    res["tritone_asc_up"] = [float(asc_up[c] / n_up[c]) for c in range(12)]
    res["tritone_asc_down"] = [float(asc_down[c] / n_down[c]) for c in range(12)]
    res["tritone_consistency"] = float(np.mean([
        asc_up[c] / n_up[c] + asc_down[c] / n_down[c] for c in range(12)]))

    print(f"[{version}] tritone ascending-rate (start pitch class):")
    print("   " + "  ".join(f"c{c}:{res['tritone_asc_up'][c]:.2f}"
                            for c in range(12)))
    print(f"[{version}] tritone consistency (up+down close to 1) = "
          f"{res['tritone_consistency']:.3f}")
    return res


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_chroma_bars(allres, save_name="shepard_chroma_both.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(12)
    width = 0.38
    for i, v in enumerate(("handy", "spikify")):
        if v not in allres:
            continue
        vals = [allres[v]["shepard_chroma_per_pitch"].get(str(c),
                allres[v]["shepard_chroma_per_pitch"].get(c, float("nan")))
                for c in range(12)]
        ax.bar(x + (i - 0.5) * width, vals, width * 0.9,
               label=f"{v}", color=("tab:blue" if v == "handy" else "tab:orange"))
    ax.axhline(CHANCE_CHROMA, color="red", ls="--", lw=1.3, label="chance 8.3%")
    ax.set_xticks(x)
    names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    ax.set_xticklabels(names)
    ax.set_ylabel("chroma accuracy (%)")
    ax.set_title("S1a: Shepard-tone chroma transfer (octave-ambiguous spectra)")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    fig.tight_layout()
    out = PLOT_DIR / save_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shepard] {save_name} saved to {out}")


def plot_register_panel(allres, envs, save_name="shepard_register_both.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    x = np.arange(8)
    width = 0.38
    for i, v in enumerate(("handy", "spikify")):
        if "reg_hist" not in allres[v]:
            continue
        h = np.asarray(allres[v]["reg_hist"])
        pct = 100.0 * h / h.sum()
        ax.bar(x + (i - 0.5) * width, pct, width * 0.9,
               label=f"{v}", color=("tab:blue" if v == "handy" else "tab:orange"))
    ax.axhline(100.0 / 8, color="red", ls="--", lw=1.3, label="uniform 12.5%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r{r}" for r in range(8)])
    ax.set_xlabel("register head argmax")
    ax.set_ylabel("share of Shepard tones (%)")
    ax.set_title("S1c: height ambiguity - register head on Shepard tones")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.6)

    ax = axes[1]
    for v in ("handy", "spikify"):
        if "register_mean_per_env" not in allres[v]:
            continue
        vals = allres[v]["register_mean_per_env"]
        ys = [vals[str(e)] for e in envs]
        ax.plot(np.log2(envs), ys, marker="o", lw=2,
                label=f"{v} (r={allres[v]['register_envcentroid_corr']:.2f})",
                color=("tab:blue" if v == "handy" else "tab:orange"))
    ax.set_xlabel("log2(spectral-envelope peak Hz)")
    ax.set_ylabel("mean register prediction")
    ax.set_title("S1c: register tracks spectral centroid (place code)?")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    out = PLOT_DIR / save_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shepard] {save_name} saved to {out}")


def plot_tritone(allres, save_name="shepard_tritone_both.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(12)
    names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    for v in ("handy", "spikify"):
        if "tritone_asc_up" not in allres[v]:
            continue
        up = allres[v]["tritone_asc_up"]
        dn = allres[v]["tritone_asc_down"]
        ax.plot(x, up, marker="o", lw=2, label=f"{v} up (c->c+6)",
                color=("tab:blue" if v == "handy" else "tab:orange"))
        ax.plot(x, dn, marker="s", lw=1.5, ls="--",
                label=f"{v} down ((c+6)->c)",
                color=("tab:blue" if v == "handy" else "tab:orange"))
    ax.axhline(0.5, color="red", ls="--", lw=1.3, label="chance 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("start pitch class")
    ax.set_ylabel("ascending-rate")
    ax.set_title("S1d: tritone paradox - directional readout per pitch class")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    out = PLOT_DIR / save_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shepard] {save_name} saved to {out}")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Shepard/tritone battery (S1)")
    p.add_argument("--version", default="handy", choices=["handy", "spikify", "both"])
    p.add_argument("--exp", default="battery", choices=["battery"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--epochs2", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=6,
                   help="Poisson re-draws per (chroma, env, phase)")
    p.add_argument("--env-centers", default="800,1600,3200",
                   help="spectral-envelope peak frequencies (Hz)")
    p.add_argument("--phases", default="cos,rand", help="comma-separated phase conditions")
    p.add_argument("--tritone-env", type=float, default=800.0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    device = torch.device(a.device)
    if a.smoke:
        a = argparse.Namespace(**{**vars(a), "epochs": 1, "epochs2": 1, "n_eval": 1,
                                  "env_centers": "800"})
    env_centers = [float(e) for e in a.env_centers.split(",")]
    phases = a.phases.split(",")
    versions = ["handy", "spikify"] if a.version == "both" else [a.version]

    allres = {}
    for version in versions:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        res = exp_battery(
            version, a.epochs, a.epochs2, a.lr, a.batch_size, device,
            a.n_eval, env_centers, phases, a.tritone_env, a.seed)
        allres[version] = res

    if len(versions) > 1:
        plot_chroma_bars(allres)
        plot_register_panel(allres, env_centers)
        plot_tritone(allres)

    out = ROOT / "data" / "shepard_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\n[shepard] results saved to {out}")


if __name__ == "__main__":
    main()
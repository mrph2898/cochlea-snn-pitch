# Cochlea-Inspired Spiking Neural Networks for Robust Pitch Perception

This repository explores the noise-robustness of temporal coding in auditory
perception using Spiking Neural Networks (SNNs). It implements a biological
auditory front-end (Gammatone filterbank + Inner Hair Cell model + Poisson
encoding) and routes the resulting spike trains into PyTorch-based SNNs built
with `snnTorch` (surrogate-gradient supervised learning and unsupervised
STDP-WTA).

Built as a course project on Neuromorphic Computing (Skoltech), and developed
further as a self-contained research repo for the Computational Neuroscience
winter school (Göttingen). The long-term goal is a bio-plausible *absolute
pitch* model: rather than a flat 88-way classifier, learn a **compressed,
octave-invariant pitch-class (chroma) representation** (see
[`docs/ABSOLUTE_PITCH.md`](docs/ABSOLUTE_PITCH.md)).

## What the project does

1. **Simulates the cochlea** — raw audio (pure tones) is passed through a
   4th-order Gammatone filterbank whose channel frequencies are spaced on
   bio-accurate Equivalent Rectangular Bandwidth (ERB) scales, then through a
   simple Inner Hair Cell (IHC) model (half-wave rectification + power-law
   compression), and encoded into spike trains via an inhomogeneous Poisson
   process.
2. **Encodes audio into spikes** with one of two interchangeable front-ends:
   - `handy` — a fully hand-rolled front-end (`src/preprocess.py`), including
     three ERB-rate scale implementations.
   - `spikify` — the `spikify` library's `FilterBank` + Poisson rate encoder.
3. **Classifies pitch with an SNN** under two learning paradigms:
   - *Supervised surrogate-gradient* training (`PitchSNN`, membrane-potential
     logits averaged over time).
   - *Unsupervised Spike-Timing-Dependent Plasticity* with Winner-Take-All
     lateral inhibition (`STDPWTA`, classical pair-based STDP + homeostatic
     weight normalization).
4. **Quantifies noise robustness** by sweeping SNR from +20 dB down to −10 dB
   and measuring classification accuracy.

## Results & conclusions

**Experimental timeline:**
1. Implemented the custom auditory preprocessing pipeline.
2. Discovered and fixed a critical **ERB scaling bug** — the Quadratic and
   Linear ERB inverse mappings returned values scaled by 1000, misplacing
   filterbank center frequencies and destroying phase-locking.
3. **3-note classification** (C4, A4, A5) — **100% accuracy** after the fix,
   for both front-ends.
4. **Full 88-key piano classification** — **~58% accuracy** with the same
   architecture.
5. **STDP unsupervised learning** — competitive WTA learning with structurally
   meaningful weight patterns.

### Handy vs. Spikify front-ends

| Property | Handy (custom) | Spikify (library) |
|----------|----------------|-------------------|
| Training speed | Slower | **Faster** |
| Noise robustness | **High (stable)** | Low (unstable at low SNR) |
| Low-frequency accuracy | **Correct from B2** | Correct from F3 |
| STDP convergence | **Stable** | Fluctuating |
| Error pattern | **Structured (octave-related)** | Diffuse |
| Temporal precision | **Better phase-locking** | Standard |

**Hypothesis:** the handy front-end's *relative thresholding and global
normalization* preserve temporal structure better, enabling robust
phase-locking to the stimulus period — the key ingredient for noise immunity
and for stable, low-frequency, octave-structured perception.

**Key open limitation:** low-frequency notes are hardest to classify (the
cochlear code degrades at low frequencies), and the flat 88-way readout is
expensive and biologically odd:

> *Learning 88 classes simultaneously is computationally expensive and possibly
> biologically implausible. Humans developing absolute pitch start from a
> limited range of pitch-class associations and generalize the octave later.*

## Toward absolute pitch (next experiments)

Replace the flat 88-class head with a **compressed, two-axis code**:

```
register r = k // 12   (0..7)        pitch class (chroma) c = k % 12
k = r * 12 + c                       # 88 notes encoded by 12 + 8 = 20 units
```

Four experiments are scaffolded in [`experiments/absolute_pitch.py`](experiments/absolute_pitch.py)
and fully described in [`docs/ABSOLUTE_PITCH.md`](docs/ABSOLUTE_PITCH.md):

1. **`chroma_full`** — train a 12-unit chroma head on all 88 notes.
2. **`chroma_transfer`** — train on one octave (C4–B4) only and measure
   octave-equivalence transfer to the other 7 registers.
3. **`chroma_register`** — separate 12+8 heads, then reconstruct the precise
   note (`k = r*12 + c`) and report 88-way accuracy.
4. **`probe`** — quantify how organized the hidden LIF population code is by
   pitch class (nearest-centroid chroma purity, t-SNE).

## Installation

This project uses `uv` for lightning-fast Python package management.

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the environment and install dependencies (Python >= 3.12):
   ```bash
   uv sync
   ```
   PyTorch is locked to the `cu132` CUDA wheel via a dedicated index in
   `pyproject.toml`.

## Pipeline overview

```
audio tone ──> [Gammatone filterbank on ERB scale] ──> [IHC: rectification + compression]
   ──> [inhomogeneous Poisson encoder] ──> 1 ms binning ──> spike tensor (Time × Channels)
   ──> SNN (snnTorch) ──> pitch / pitch-class / register
```

### Auditory front-end (`src/preprocess.py`)

- **`generate_tone` / `add_white_noise`** — synthesizes sine tones and
  corrupts them with Gaussian white noise at a controllable SNR.
- **ERB-rate scales** — `LinearERB` (Glasberg & Moore, 1990), `QuadERB`, and
  `VoiceboxERB` (Moore & Glasberg style), all invertible, used to place
  filterbank center frequencies on a perceptually uniform axis
  (`erb_space`). Validated by a round-trip test (`inv(call(f)) ≈ f`).
- **`gammatone_filterbank`** — FIR Gammatone filters
  (`h(t) = t^(n-1) e^(-2π b t) cos(2π f t)`), default 50 channels, 100 Hz–6.5 kHz,
  ERB-derived bandwidths (`b = 1.019·ERB(f)`), applied via FFT convolution.
- **`ihc_rectification_compression`** — soft half-wave rectification followed
  by power-law compression (`x^0.7`) to mimic IHC transduction.
- **`generate_poisson_spikes`** — inhomogeneous Poisson spike trains whose
  instantaneous rate is proportional to the filtered IHC output, scaled
  against a *global* maximum so relative loudness across channels is preserved
  (high-energy channels fire near the 800 Hz peak rate).
- **Visualization** — spike rasters with the y-axis mapped to channel CF,
  clean-vs-noisy "synchronized cochlear response" figures (showing
  phase-locking / volley patterns), and a comparison of the ERB scales.

### SNN architectures

- **`PitchSNN`** (`src/snn.py`) — feed-forward SNN:
  `Linear(50→100) + Leaky-LIF` → `Linear(100→C) + Leaky (threshold=1e9)`. The
  final layer is a non-spiking leaky integrator whose membrane potential over
  time serves as logits; the time-averaged membrane is fed to cross-entropy,
  trained with surrogate-gradient descent (fast-sigmoid).
- **`STDPWTA`** (`src/stdp.py`) — fully unsupervised layer with LIF output
  neurons, lateral WTA inhibition, manual classical pair-based STDP
  (pre/post eligibility traces with exponential decay, τ = 20 ms, all-to-all,
  applied only to the winning neuron), and homeostatic L2 normalization.

### Dataset generation (`src/generate_piano_dataset.py`)

- Synthesizes the full **88-key piano range (A0 = 27.5 Hz → C8 = 4186 Hz)**,
  50 samples/key, with mild randomized noise (10–30 dB SNR) for robustness.
- Raw 44.1 kHz spikes are temporally binned into **1 ms bins** (any spike in
  the window → 1), collapsing 4410 steps down to 100.
- Two dataset versions, one per front-end: `data/piano_dataset_handy.pt` and
  `data/piano_dataset_spikify.pt`.

### Experiments

- **`experiments/train_and_evaluate.py`** — supervised training in two regimes:
  `3classes` (C4/A4/A5, per-SNR evaluation, *Accuracy vs. SNR* plot) and
  `full_piano` (88 classes, test accuracy + confusion matrix).
- **`src/stdp.py`** (run as a script) — unsupervised STDP-WTA training on the
  piano datasets, winner selection by highest spike count, followed by a
  majority-vote neuron-to-class assignment for evaluation.
- **`experiments/advanced_train.py`** — an alternative front-end using
  `spikify`'s `DeltaModulator` (threshold-crossing up/down encoding) at 16 kHz,
  using the *final* membrane state (not averaged) as logits.
- **`experiments/absolute_pitch.py`** — the octave-invariant chroma experiments
  (see above and `docs/ABSOLUTE_PITCH.md`).

## Artifacts

Stored under `data/`:

- **Models**: `model_piano88_*.pth` (supervised, both front-ends),
  `model_stdp_wta_*.pth` (unsupervised, incl. a 20-epoch run), plus STDP
  `neuron_assignments` files.
- **Datasets**: `piano_dataset_handy.pt`, `piano_dataset_spikify.pt`.
- **Plots**:
  - `training_curves_piano88_{frontend}.pdf` — loss/accuracy curves.
  - `stdp_spike_count_{frontend}.pdf` — average spikes/sample over STDP epochs.
  - `confusion_stdp_*.pdf`, `confusion_piano88_*.pdf` — 88×88 confusion
    matrices with note-name labels (black keys grey).
  - `filterbank_freq*.pdf` — clean-vs-noisy cochlear rasters for the hand-rolled
    and library front-ends.
  - `erb_comparison.pdf` — comparison of the ERB-rate scales.
  - `noise_robustness.png` — SNR sweep of the supervised 3-class regime.

## Repository layout

```
├── README.md                         # this file
├── pyproject.toml                    # uv/python project metadata
├── src/                              # library package
│   ├── preprocess.py                 # cochlear front-end (ERB, gammatone, IHC, Poisson)
│   ├── snn.py                        # supervised PitchSNN architecture
│   ├── stdp.py                       # unsupervised STDP-WTA layer + training/eval
│   └── generate_piano_dataset.py     # 88-key spike dataset generation
├── experiments/                      # runnable experiment scripts
│   ├── train_and_evaluate.py         # supervised 3-class / 88-class + noise robustness
│   ├── advanced_train.py             # delta-modulator front-end experiment
│   └── absolute_pitch.py             # octave-invariant chroma/register experiments
├── docs/
│   └── ABSOLUTE_PITCH.md             # absolute-pitch experiment design
└── data/                             # generated artifacts (datasets, models, plots)
```

## Usage

```bash
# 1. Generate the 88-key spike datasets (handy / spikify variants)
uv run python -m src.generate_piano_dataset    # change version=... inside

# 2. Train and evaluate the supervised SNN (3-class or full piano)
uv run python experiments/train_and_evaluate.py   # change main(...) inside

# 3. Train the unsupervised STDP-WTA network and evaluate
uv run python -m src.stdp

# 4. Alternative front-end experiment (spikify delta modulators)
uv run python experiments/advanced_train.py

# 5. Absolute-pitch experiments (octave-invariant chroma compression)
uv run python experiments/absolute_pitch.py --version handy --exp chroma_transfer --epochs 20
uv run python experiments/absolute_pitch.py --smoke   # quick pipeline check
```

## References

- Glasberg, B. R. & Moore, B. C. J. (1990). Derivation of auditory filter
  shapes from notched-noise data. *Hearing Research.*
- Moore, B. C. J. & Glasberg, B. R. (1983). Suggested formulae for calculating
  auditory-filter bandwidths and excitation patterns. *J. Acoust. Soc. Am.*
- Patterson, R. D. et al. (1995). Complex sounds and auditory images. In
  *Auditory Physiology and Perception* (Gammatone filterbank formulation).
- Cariani, P. (1999). Temporal coding of periodicity pitch in the auditory
  system. *Neural Plasticity.*
- Eshraghian, J. K. et al. (2021). Training spiking neural networks using
  lessons from deep learning (snnTorch). arXiv:2105.06696.
- Pan, Z., et al. (2019). An efficient and perceptually motivated auditory
  neural encoding algorithm for SNNs. *IEEE TNNLS.*
- Song, Z., et al. (2024). Spiking-LEAF: A learnable auditory front-end for
  spiking neural networks. *IEEE TPAMI.*
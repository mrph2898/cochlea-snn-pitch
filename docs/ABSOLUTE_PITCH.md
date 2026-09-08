# Toward an "Absolute Pitch" Model: Octave-Invariant Chroma Compression

## Motivation

The 88-key classification task (see presentation, *Experiment 2*) reached only
~58% accuracy. A flat 88-way readout forces the network to jointly learn:

1. **Absolute frequency discrimination** within an octave,
2. **Octave equivalence** (C4 vs. C5 share a pitch class),
3. **Harmonic structure** across 7+ octaves.

From a biological standpoint this is a strange way for the auditory system to
solve pitch. Human cases of *perfect/absolute pitch* typically develop by
learning **pitch-class associations within a limited range** and later
**generalizing octave-invariant pitch classes** to the whole keyboard. This
project therefore reformulates the problem as **information compression**:
represent each note as the Cartesian product of a *pitch class* (chroma) and a
*register*:

```
note index  k = 0..87
register    r = k // 12     (0..7,  the octave band)
pitch class c = k % 12      (chroma, octave-invariant)

k = r * 12 + c              # 88 keys encoded by 12 + 8 = 20 units
```

## Hypotheses

- **H1 (compression).** Learning a 12-unit chroma representation across the
  full 88-note range reaches higher accuracy than the flat 88-way model for
  the *same* model sizes, because the task transfers knowledge across octaves.
- **H2 (octave equivalence).** A chroma head trained on a *single octave*
  (C4–B4) transfers to other octaves with above-chance accuracy without any
  retraining — i.e., octave-invariant features emerge from the cochlear code.
- **H3 (reconstruction).** Independent chroma (12) + register (8) heads can
  reconstruct exact notes (`k = r*12 + c`) with 88-way accuracy above the
  chance level of 1.1%, proving the compressed code carries full pitch info.
- **H4 (representational structure).** Hidden LIF population activity becomes
  organized by pitch class (assessed via nearest-centroid chroma purity and
  t-SNE), even when trained only on one octave.

## Experiments

Implemented as `experiments/absolute_pitch.py`:

| # | Experiment | Protocol | Metric |
|---|------------|----------|--------|
| 1 | `chroma_full` | Train 12-unit chroma head on **all** 88 notes (80/20 split; identical SNN/SG settings as 88-way baseline) | chroma test acc vs. 8.3% chance; compare against 88-way of the same architecture |
| 2 | `chroma_transfer` | Train 12-unit chroma head **only on register 3 (C4–B4)**; then probe all notes | per-register chroma acc → curve of octave-equivalence transfer |
| 3 | `chroma_register` | Train chroma head (12) + register head (8) separately; combine on the test split | chroma %, register %, reconstructed 88-note accuracy |
| 4 | `probe` | Record hidden LIF spike counts during inference | nearest-centroid chroma purity, t-SNE of the population code colored by chroma |

All experiments run for the `handy` and `spikify` datasets to keep the
front-end comparison.

## Results: handy vs spikify (20 epochs, Adam lr=2e-4, seed 0)

Full suite (`--version both --exp suite`): all four experiments on both
front-ends. Metrics in `data/absolute_pitch_results.json`, comparison figures
in `data/plots/absolute_pitch_{transfer,summary}_both.png` and
`data/plots/tsne_chroma_both.png`.

| Experiment | Chance | Handy | Spikify |
|-----------|--------|-------|---------|
| `chroma_full` (12 units, all 88 notes) | 8.3% | **46.3%** | **43.3%** |
| `chroma_transfer` held-out (same octave, reg 3) | 8.3% | 14.2% | **33.3%** |
| `chroma_transfer` other registers (0–2, 4–7) | 8.3% | ~chance (1–9%, reg 7: 25% on 4 classes = chance) | ~chance (5–15%, reg 7: 26.5% = chance) |
| `chroma_register`: chroma head / register head / reconstructed note | 8.3 / 12.5 / 1.1% | 45.6 / **95.1** / **44.9%** | 43.4 / **99.4** / **43.4%** |
| `probe`: hidden-code chroma purity | 8.3% | **34.8%** | 15.8% |

**Interpretation.**
1. **Compression works for both front-ends.** A 20-unit chroma+register code
   reconstructs the exact 88-key note at ~44% (chance 1.1%) — near the flat
   88-way model's ~57% at 1/4 of the output size. Register (absolute height)
   is nearly trivial (95–99%); pitch class is the bottleneck (~44%).
2. **No zero-shot octave transfer.** A chroma head trained on a single octave
   does *not* generalize to other octaves — every non-trained register sits at
   chance for *both* front-ends. The network binds pitch class to the absolute
   tonotopic location of the trained octave; H2 in its naive form is
   **falsified**, and an explicit octave-invariant mechanism (e.g. tonotopic
   shift augmentation) is needed.
3. **Spikify learns the trained distribution faster; handy builds more
   structured codes.** Spikify wins the same-octave held-out test
   (33.3% vs 14.2% — faster convergence, as in the course results), while
   handy's hidden population code is more than twice as chroma-organized
   (purity 34.8% vs 15.8%) — consistent with the thesis that its relative
   thresholding preserves temporal structure.
4. **Chroma *is* learnable across octaves jointly.** `chroma_full` reaches
   ~45% when all octaves are shown together: the network learns a shared
   pitch-class code from data, it just does not abstract it from one octave
   alone.

**Notes on methodology.**
- Splits must be *shuffled*: the dataset is stored blocked by note (50
  samples per key), so a contiguous 80/20 split isolates octaves.
- `per_register_accuracy` must index *sample positions*, not note ids —
  an early version of this bug produced phantom 83–100% "upward transfer".
  Always sanity-check per-register curves against the in-split held-out score.

## Implementation notes

- Labels: `register = k // 12`, `chroma = k % 12` are exact for the A0..C8
  layout (verified against `piano_key_labels()`).
- Model: the same `PitchSNN` (50 → 100 LIF → C) with surrogate-gradient
  training on time-averaged membrane logits. `ProbePitchSNN` additionally
  records hidden spikes for the representation analysis without changing the
  dynamics.
- Training: Adam (lr ≈ 2e-4), 64-mini-batches over the 1 ms-binned spike
  tensors (100 time steps), 20 epochs per head.

## Expected analysis / report items

- Accuracy vs. number-of-output-units (88 → 12 → 20), demonstrating compression.
- Per-register transfer curves: flat line ⇒ learned pitch class is octave
  invariant; decaying line ⇒ the network still makes octave-specific mistakes.
- t-SNE plots of the hidden population code, colored by chroma.
- Reconstruction confusion matrix over all 88 notes from the 20-unit code.
- Discussion connecting the results to Cariani (1999) temporal-coding of
  periodicity pitch and the "place vs. temporal" coding trade-off.

## Follow-up ideas (sketch)

- **Channel-shift octave augmentation:** because gammatone CFs are log-spaced,
  an octave is a constant shift along the tonotopic axis; augment training by
  spatially shifting the cochlear-channel map, forcing translation invariance.
- **Latency vs. rate coding probe:** measure which cochlear channels the chroma
  head relies on (e.g., via input-gradients) to distinguish place-based
  (CF-aligned) and temporal (phase-locked low-CF) strategies.
- **STDP chroma layer:** run the STDP-WTA layer with 12 output neurons on chroma
  labels and compare structural selectivity with the supervised variant.
- **Weight-dependent ("long-term") plasticity:** extend the additive STDP
  update with multiplicative/weight-dependent terms so weight changes saturate,
  enabling long-term memory of pitch-class prototypes (see the course framing
  on long-term plasticity models).
# Literature Review: Biological Grounding of the Neuromorphic Cochlea Project

Scope: how each component of this repository's pipeline — gammatone
filterbank on ERB scales, IHC rectification/compression, Poisson spike
encoding, LIF readout, STDP-WTA — maps onto the mammalian auditory system;
how the experiments (noise robustness, 88-key classification, chroma/register
compression) map onto auditory neuroscience; and which *modern* hypotheses the
model is now positioned to test. All cited works below were verified against
primary sources during this review (September 2026).

---

## Part I — Biological grounding of the pipeline

### I.1. Basilar-membrane filtering → gammatone filterbank (`gammatone_filterbank`)

The cochlea performs a mechanical frequency analysis: each place along the
basilar membrane (BM) responds maximally to one characteristic frequency (CF),
producing the tonotopic map preserved up to primary auditory cortex. The
gammatone function was introduced to summarize reverse-correlation ("revcor")
measurements of auditory-nerve impulse responses (Johannesma, 1972), and an
efficient gammatone auditory filterbank was developed as the spectral-analysis
stage of peripheral models (Patterson, Nimmo-Smith, Holdsworth & Rice,
1987/1988). Patterson et al. (1992) showed that a 4th-order gammatone gives an
excellent fit to human auditory-filter shapes; Slaney (1993) published the
reference digital implementation ("Patterson's ear"), and Hohmann (2002)
provided the widely used all-pole approximation. Patterson's Auditory Image
Model (AIM) chains exactly our two stages — gammatone filterbank (BM motion)
followed by a transduction/quantization stage into a multi-channel "neural
activity pattern" (Patterson & Holdsworth, 1996; Patterson, 2000).

*Mapping to this repo:* `gammatone_filterbank` (order 4, `b = 1.019·ERB(f)`,
FFT convolution, 50 channels) is a standard BM-motion simulation; each output
channel stands in for ~0.9 mm of membrane driving one IHC/auditory-nerve
population.

### I.2. Auditory-filter bandwidths → ERB scales (`LinearERB/QuadERB/VoiceboxERB`)

Frequency selectivity was first quantified as "critical bands" (Fletcher,
1940) and parametrized with rounded-exponential (roex) filter shapes fitted to
notched-noise masking data (Patterson & Moore, 1986). Glasberg & Moore (1990,
*Hearing Research* 47:103–138) summarized young normal-hearing data as
`ERB(F) = 24.7·(4.37F + 1)` Hz and the ERB-rate scale
`ERBS(f) = 21.4·log10(1 + 4.37f)`; an earlier quadratic approximation,
`ERB(F) = 6.23F² + 93.39F + 28.52`, comes from Moore & Glasberg (1983) — the
two non-linear scales implemented here as `QuadERB`/`VoiceboxERB`. One ERB
step corresponds to roughly constant distance (~0.9 mm) along the BM, i.e. the
scale is the natural coordinate for placing filter CFs (the `erb_space`
routine). The ×1000 bug found in this project is instructive precisely
because biology is unforgiving about this coordinate: mis-scaled CFs destroy
the tonotopic code.

### I.3. Transduction → IHC rectification + compression (`ihc_rectification_compression`)

Information reaches the auditory nerve almost exclusively through inner hair
cells; outer hair cells (OHCs) serve mainly as the cochlear amplifier
(prestin-based electromotility). The canonical computational IHC is Meddis's
transmitter-release model and its revision (Sumner, Lopez-Poveda, O'Mard &
Meddis, 2002, *JASA* 111:2178–2188; Sumner, 2003), which reproduces
adaptation, saturation and rate-level functions via limited
transmitter supply/recycling at the ribbon synapse (see also Zhang & Carney,
2005; Bruce, Erfani & Zilany, 2018). Short-term firing-rate adaptation is
attributed primarily to presynaptic vesicle depletion (Wen et al., 2009,
*J. Neurosci.* 29:13797), and dynamic-range adaptation to sound-level
statistics likewise localizes to the IHC synapse rather than mechanics.

*Mapping to this repo:* half-wave rectification + `x^0.7` power-law
compression is a deliberately minimal IHC: rectification captures the
unidirectional (depolarizing-phase) drive of the ribbon synapse, and the
exponent mimics compressive loudness growth. Missing relative to biology:
transmitter-depletion adaptation, refractoriness, spontaneous-rate fiber
classes, OHC active gain, and efferent (MOC/LOC) feedback.

### I.4. Auditory-nerve firing → Poisson encoder (`generate_poisson_spikes`)

Auditory-nerve spikes phase-lock to the stimulus fine structure up to
~4–5 kHz (the "volley principle": fibers take turns firing on successive
cycles), which is the physiological basis of every temporal pitch theory;
above that limit only place cues remain (Oxenham et al., 2011, *PNAS*;
Moore & Ernst, 2012). Phenomenological AN models (Zilany, Bruce & Carney,
2009, *JASA*; updated 2014) formalize the full periphery
(BM → IHC → synapse → stochastic spikes) and are the gold standard this
project's encoder approximates.

*Mapping to this repo:* the inhomogeneous-Poisson encoder with global-max
rate normalization reproduces two key facts — firing probability tracks the
IHC drive cycle-by-cycle (hence the visible phase-locking/volley rasters),
and relative level across channels is preserved. Missing: refractoriness,
phase-locking roll-off with CF, and level-dependent synchrony capture.

### I.5. Readout neurons → LIF (`PitchSNN`)

Leaky integrate-and-fire dynamics with lateral inhibition are the standard
abstraction of central auditory neurons (e.g. cochlear-nucleus stellate/chopper
responses are already captured by LIF + dendritic filtering — Hewitt, Meddis
& Shackleton, 1992). The surrogate-gradient training itself is *not*
biologically claimed here; it is an engineering tool to test what the
cochlear code linearly affords.

### I.6. Unsupervised plasticity → STDP-WTA (`STDPWTA`)

Classical STDP (pre-before-post → LTP, post-before-pre → LTD on ~10 ms
scales: Markram et al., 1997; Bi & Poo, 1998; Song et al., 2000; reviewed in
Caporale & Dan, 2008; Feldman, 2012, *Neuron*) was in fact *predicted* for the
auditory system (Gerstner et al., 1996, development of phase locking in sound
localization) and then demonstrated in it: repetitive pairing of tones of
different frequencies shifts frequency tuning of ferret primary-auditory-cortex
neurons with STDP-like temporal specificity (Dahmen et al., 2008,
*J. Neurosci.* 28:13629), and STDP-like protocols alter auditory perceptual
learning (Mossbridge et al., 2006). Developmentally, spontaneous Ca²⁺ activity
in pre-hearing IHCs instructs tonotopic-map formation (Kros et al., 1998;
reviewed in Johnson & Marcotti literature; Kandler et al., 2009), and STDP
forms switch from order-independent (all-LTP, network formation) to Hebbian
(refinement) during critical periods.

*Mapping to this repo:* the WTA layer with lateral inhibition + homeostatic
normalization is a functional stand-in for competitive map refinement; its
structured (octave-related) weight patterns mirror how timing-driven
plasticity can carve frequency-selective populations without supervision.

---

## Part II — How the experiments map onto auditory neuroscience

### II.1. Noise robustness ↔ the temporal-coding hypothesis

The project's core hypothesis (temporal coincidence > rates in noise) sits
inside the central century-old debate: place (Helmholtz/Wightman/Terhardt
harmonic templates) vs. temporal (Licklider, 1951, autocorrelation; Cariani
& Delgutte, 1996, AN interspike intervals; Meddis & O'Mard, 1997, unitary
temporal model) vs. combined place–time codes (Shamma & Klein, 2000;
Cedolin & Delgutte, 2010; reviewed in Oxenham, 2012, *J. Neurosci.*;
Oxenham, 2013; Oxenham, 2018, *Ann. Rev. Psych.*; Oxenham, 2022–2023,
*Front. Neurosci.*). Key touchstones for our SNR curves:
resolved harmonics give salient pitch via place, unresolved harmonics via
envelope timing (Bernstein & Oxenham, 2003); temporal codes fail above the
phase-locking limit (Oxenham et al., 2011) and pitch transitions from
temporal to place at high frequencies (Moore & Ernst, 2012); cochlear-implant
listeners — place-degraded, timing-mediated — show exactly the fragile,
envelope-like pitch our spikify-at-low-SNR behavior resembles. Our finding
(handy degrades gracefully, spikify collapses) is therefore a concrete,
model-level instance of the claim that *preserved fine timing buys noise
immunity*.

### II.2. Octave confusions ↔ the chroma/height dissociation

Bachem's classic distinction splits pitch into circular **chroma** (pitch
class, repeating each octave) and linear **height**. Cortex respects it:
pitch height activates posterior planum temporale while chroma engages
planum polare, anterior to A1 — a hierarchical stream with specialized roles
beyond A1 (Warren et al., 2003). Our 88-way model's error pattern (mistakes
almost exclusively *within* chroma) is the computational analogue: the
network discovers height easily and chroma with effort, exactly the
decomposition we then imposed explicitly (register + chroma heads at
95–99% vs ~44%). Recent MEG work further shows height and chroma are
*temporally dissociable* cortical representations, with slow (delta-band)
activity encoding their relative relationships (Abrams et al., 2025,
*J. Neurosci.*) — a prediction our two-head architecture is well placed to
operationalize (e.g. fast vs. slow readout timescales).

### II.3. Single-octave training ↔ absolute-pitch development

Human absolute pitch (AP) is best described by a two-component model —
long-term pitch *memory* plus verbal *labeling* (Levitin, 1994; Levitin &
Rogers, 2005) — emerging from genes × early training inside a sensitive
period (Zatorre, 2003, *Nat. Neurosci.*; Deutsch et al., 2006), with
anatomical markers (planum-temporale asymmetry — Schlaug et al., 1995;
broader tuning in enlarged auditory cortex — McKetton et al., 2019) and a
fronto-temporal labeling network (Zatorre et al., 1998, *PNAS*). Our
"train on C4–B4, generalize by octave" protocol is the developmental
trajectory minus the label-association stage — and its *failure* (no
zero-shot transfer) is informative: bare tonotopic exposure does not induce
octave equivalence, consistent with AP requiring associative/labeling
machinery on top of sensory maps. The next step is therefore to give the
model that second component (see H-D/E below).

### II.4. STDP experiment ↔ auditory cortical plasticity

Dahmen et al. (2008) is the direct biological counterpart: millisecond-scale
tone pairing reshapes A1 frequency tuning STDP-style. Our WTA layer shows the
complementary computational fact — competitive timing-driven plasticity alone
suffices to form pitch-selective populations from cochlear spike trains, with
selectivity quality inherited from front-end timing fidelity.

---

## Part III — Modern hypotheses this model can now test

Each item names the hypothesis, the protocol sketch (all implementable in
`experiments/`), and the decisive figure.

- **H-A — Chroma/height dissociation with Shepard tones.** Stimuli that fix
  chroma while making height ambiguous (octave-spaced components under a
  fixed bell envelope — Shepard, 1964, *JASA* 36:2346; Risset, 1971) should be
  classified correctly by the chroma head while the register head sits at
  chance; tritone pairs (Deutsch, 1986; 1992; 2008) probe whether the model
  shows a systematic "higher/lower" bias by pitch class — the SNN analogue of
  the tritone paradox and of latent absolute pitch in non-possessors.
- **H-B — Resolved vs. unresolved harmonics.** Feed harmonic complexes with
  and without the fundamental, and with harmonics confined above/below the
  phase-locking limit (Bernstein & Oxenham, 2003; Oxenham et al., 2011;
  Moore & Ernst, 2012): a temporal-code model must show missing-F0 pitch and
  a high-frequency breakdown; comparing handy vs spikify quantifies how much
  of each survives each encoder.
- **H-C — What would a learnable cochlea discover?** Make CFs, bandwidths and
  the compression exponent learnable and train jointly on the chroma task
  (cf. LEAF — Zeghidour et al., 2021, *ICLR*; Spiking-LEAF with its IHC-LIF
  two-compartment encoder and lateral feedback — Song et al., 2024,
  *ICASSP*; cautionary analysis in Schlüter & Gutenbrunner, 2022, and Meng
  et al., 2023). Decisive test: does learning re-discover ERB spacing, or
  something task-specifically different?
- **H-D — Warren-2003-style attribution.** Occlude/lesion channel groups and
  measure per-head accuracy drops: prediction is register←high-CF place cues,
  chroma←low-CF phase-locked timing. This turns the [TODO] attribution item
  into a test of the cortical height/chroma stream dissociation.
- **H-E — Contextual pitch (modern predictive view).** Precede ambiguous
  (Shepard/tritone) probes with biasing contexts and test for
  expectation-driven shifts of the decoded pitch (cf. context effects down to
  A1 — Englitz et al., 2024; sensorimotor predictive representations —
  Abrams/Gwilliams line of work, 2024–2025).
- **H-F — Sensitive-period curriculum.** Compare full-range, narrow→wide
  staged, and STDP-pretrained-then-supervised training on chroma accuracy
  *and* on zero-shot transfer — the computational test of Zatorre's (2003)
  genes×training×timing account and of the developmental STDP-form switch.
- **H-G — Hearing-loss / CI degradation.** Broaden filters or drop channels
  (vocoder-style) and re-run the chroma battery: predicts selective chroma
  collapse mirroring human hearing-loss effects on pitch (Bernstein &
  Oxenham, 2006) and extends our SNR robustness story into spectral
  degradation.
- **H-H — Binaural pitch in noise.** Add a second ear + MSO/LSO-style
  coincidence stage; test binaural unmasking of chroma — the natural sequel
  to the monaural noise-robustness result.

## Part IV — Prioritized next experiments (concrete)

1. **S1 — Shepard/tritone battery (H-A).** New `experiments/shepard_probe.py`:
   synthesize octave-complex tones (fixed Gaussian envelope on log-f axis),
   run trained chroma+register heads, report chroma acc vs register-head
   entropy + tritone direction-bias curves. Figures → `data/plots/`.
2. **H1 — Harmonic-complex battery (H-B).** Extend dataset generation with
   missing-F0 and resolved/unresolved complexes; evaluate frozen chroma heads.
3. **L1 — Learnable front-end (H-C).** Differentiable CF/bandwidth/compression
   parameters trained jointly on chroma; compare learned vs ERB maps.
4. **A1 — Head attribution (H-D).** Gradient/occlusion importance of channels
   per head; low-CF vs high-CF reliance statistics.
5. **Remaining README TODOs in order:** longer chroma schedule → low-frequency
   filterbank density → tonotopic-shift augmentation (re-test zero-shot
   transfer — the direct follow-up to the falsified H2) → STDP chroma layer →
   weight-dependent STDP → binaural extension.

---

## Bibliography

- Abrams, E. B. et al. (2025). Temporally dissociable neural representations
  of pitch height and chroma. *J. Neurosci.* 45(8).
- Baek, S. (2024). SNN and sound: a comprehensive review of spiking neural
  networks in sound. *Biomed. Eng. Lett.* (PMC11362401).
- Bernstein, J. G. & Oxenham, A. J. (2003). Pitch discrimination of diotic and
  dichotic tone complexes. *JASA* 113:3323–3334.
- Bernstein, J. G. & Oxenham, A. J. (2006). The effect of hearing loss on
  pitch perception. *Hear. Res.* (series).
- Bi, G.-q. & Poo, M.-m. (1998). Synaptic modifications in cultured
  hippocampal neurons. *J. Neurosci.* 18:10464–10472.
- Bruce, I. C., Erfani, Y. & Zilany, M. S. A. (2018). A phenomenological
  model of the IHC–AN synapse. *Hear. Res.* 360:40–54.
- Caporale, N. & Dan, Y. (2008). Spike timing–dependent plasticity: a Hebbian
  learning rule. *Ann. Rev. Neurosci.* 31:25–46.
- Cariani, P. A. & Delgutte, B. (1996). Neural correlates of the pitch of
  complex tones. *J. Neurophysiol.* 76:1698–1716.
- Cedolin, L. & Delgutte, B. (2010). Spatiotemporal representation of the
  pitch of harmonic complex tones in the auditory nerve. *J. Neurosci.*
  30:127–143.
- Dahmen, J. C. et al. (2008). Stimulus-timing-dependent plasticity of
  cortical frequency representation. *J. Neurosci.* 28:13629–13639.
- de Cheveigné, A. (2010). Pitch perception. In *Oxford Handbook of Auditory
  Science: Hearing*.
- Deutsch, D. (1986). A musical paradox. *Music Perception* 3:275–280.
- Deutsch, D. (1992). Paradoxes of musical pitch. *Scientific American*
  267:88–95.
- Deutsch, D., Dooley, K. & Henthorn, T. (2008). Pitch circularity from tones
  comprising full harmonic series. *JASA* 124:589–597.
- Deutsch, D. et al. (2006). Absolute pitch among American and Chinese
  conservatory students. *JASA* 119:719–722.
- Eshraghian, J. K. et al. (2023). Training spiking neural networks using
  lessons from deep learning. *Proc. IEEE* 111(9). (snnTorch; preprinted
  2021, arXiv:2105.06696.)
- Feldman, D. E. (2012). The spike-timing dependence of plasticity.
  *Neuron* 75:556–571.
- Fletcher, H. (1940). Auditory patterns. *Rev. Mod. Phys.* 12:47–65.
- Gerstner, W. et al. (1996). A neuronal learning rule for sub-millisecond
  temporal coding. *Nature* 383:76–78.
- Glasberg, B. R. & Moore, B. C. J. (1990). Derivation of auditory filter
  shapes from notched-noise data. *Hear. Res.* 47:103–138.
- Greenwood, D. D. (1990). A cochlear frequency-position function for several
  species. *JASA* 87:2592–2605.
- Hewitt, M. J., Meddis, R. & Shackleton, T. M. (1992). A computer model of
  a cochlear-nucleus stellate cell. *JASA* 91:2096–2109.
- Hohmann, V. (2002). Frequency analysis and synthesis using a gammatone
  filterbank. *Acta Acustica* 88:433–442.
- Johannesma, P. I. M. (1972). The pre-response stimulus ensemble of neurons
  in the cochlear nucleus. *IPO Symposium on Hearing Theory*.
- Lau, B. K., Mehta, A. H. & Oxenham, A. J. (2017). Superoptimal perceptual
  integration suggests a place-based representation of pitch at high
  frequencies. *J. Neurosci.* 37:9013–9021.
- Levitin, D. J. (1994). Absolute memory for musical pitch. *Perception &
  Psychophysics* 56:414–423.
- Levitin, D. J. & Rogers, S. E. (2005). Absolute pitch: perception, coding,
  and controversies. *Trends Cogn. Sci.* 9:26–33.
- Licklider, J. C. R. (1951). A duplex theory of pitch perception.
  *Experientia* 7:128–134.
- Markram, H. et al. (1997). Regulation of synaptic efficacy by coincidence
  of postsynaptic APs and EPSPs. *Science* 275:213–215.
- McKetton, L., DeSimone, K. & Schneider, K. A. (2019). Larger auditory
  cortical area and broader frequency tuning underlie absolute pitch.
  *J. Neurosci.* 39:2930–2937.
- Meddis, R. & O'Mard, L. (1997). A unitary model of pitch perception.
  *JASA* 102:1811–1820.
- Meng, H. et al. (2023). What is learnt by the LEArnable Front-end (LEAF)?
  *Interspeech 2023*.
- Moore, B. C. J. & Ernst, S. M. (2012). Frequency difference limens at high
  frequencies: evidence for a transition from a temporal to a place code.
  *JASA* 132:1542–1547.
- Moore, B. C. J. & Glasberg, B. R. (1983). Suggested formulae for calculating
  auditory-filter bandwidths and excitation patterns. *JASA* 73:750–753.
- Mossbridge, J. A. et al. (2006). Perceptual-learning evidence for
  spike-timing-dependent plasticity in the human auditory system.
  *Neuroreport*-series work cited in Dahmen et al. (2008).
- Oxenham, A. J. (2012). Pitch perception. *J. Neurosci.* 32:13335–13338.
- Oxenham, A. J. (2013). Revisiting place and temporal theories of pitch.
  *Hear. Res.* (Frontiers-style review; PMC4215732).
- Oxenham, A. J. (2018). How we hear: the perception and neural coding of
  sound. *Ann. Rev. Psych.* 69:27–50.
- Oxenham, A. J. (2022). Questions and controversies surrounding the
  perception and neural coding of pitch. *Front. Neurosci.* 16:1074752.
- Oxenham, A. J. et al. (2011). Pitch perception beyond the traditional
  existence region of pitch. *PNAS* 108:7629–7634.
- Pan, Z. et al. (2019). An efficient and perceptually motivated auditory
  neural encoding algorithm for SNNs. *IEEE TNNLS*.
- Patterson, R. D. (2000). Auditory Image Model (AIM): how complex sounds are
  represented in the auditory system (MRC-CBU exposition).
- Patterson, R. D. & Holdsworth, J. (1996). A functional model of neural
  activity patterns and auditory images. *Adv. Speech Hear. Lang. Proc.* 3B.
- Patterson, R. D. & Moore, B. C. J. (1986). Auditory filters and excitation
  patterns. In Moore (Ed.), *Frequency Selectivity in Hearing*.
- Patterson, R. D., Nimmo-Smith, I., Holdsworth, J. & Rice, P. (1987/1988).
  An efficient auditory filterbank based on the gammatone function (SVOS
  report; APU Tech. Note).
- Patterson, R. D. et al. (1992). Complex sounds and auditory images. *Proc.
  9th Int. Symp. Hearing*.
- Risset, J.-C. (1971). Paradoxes de hauteur. *Proc. 7th ICA*, Budapest.
- Saddler, M. R., Gonzalez, R. & McDermott, J. H. (2021). Deep neural network
  models reveal interplay of peripheral coding and stimulus statistics in
  pitch perception. *Nature Commun.* 12:7278. (Key result: AN phase locking
  is critical for human-like pitch; training *with* noise and natural sounds
  is required — an "auditory diet" effect directly relevant to our
  noise-injected training regime.)
- Schlaug, G. et al. (1995). Increased corpus callosum size / planum
  temporale asymmetry in musicians with absolute pitch. *Science*
  267:699–701 (PT asymmetry result).
- Schlüter, J. & Gutenbrunner, G. (2022). EfficientLEAF: a faster LEarnable
  audio frontend of questionable use. *arXiv:2207.05508*.
- Shamma, S. & Klein, D. (2000). The case of the missing pitch templates.
  *JASA* 107:2631–2644.
- Shepard, R. N. (1964). Circularity in judgments of relative pitch.
  *JASA* 36:2346–2353.
- Slaney, M. (1993). An efficient implementation of the Patterson–Holdsworth
  auditory filter bank. *Apple Computer Tech. Rep.* 35.
- Song, S. et al. (2000). Competitive Hebbian learning through
  spike-timing-dependent synaptic plasticity. *Nature Neurosci.* 3:919–926.
- Song, Z. et al. (2024). Spiking-LEAF: a learnable auditory front-end for
  spiking neural networks. *ICASSP 2024* (arXiv:2309.09469).
- Sumner, C. J. et al. (2002). A revised model of the inner-hair cell and
  auditory-nerve complex. *JASA* 111:2178–2188.
- Sumner, C. J. (2003). Adaptation in a revised inner-hair cell model
  (PhD-series work; PMID 12597183).
- Terhardt, E. (1974). Pitch, consonance, and harmony. *JASA* 55:1061–1069.
- Warren, J. D., Uppenkamp, S., Patterson, R. D. & Griffiths, T. D. (2003).
  Separating pitch chroma and pitch height in the human brain. *PNAS*
  100:10038–10042 (height→posterior planum temporale; chroma→planum
  polare, anterior to Heschl's gyrus).
- Wen, B. et al. (2009). Dynamic range adaptation to sound level statistics
  in the auditory nerve. *J. Neurosci.* 29:13797–13808.
- Wu, J. et al. (2018). A spiking neural network framework for robust sound
  classification. *Front. Neurosci.* 12:127.
- Zatorre, R. J. (2003). Absolute pitch: a model for understanding the
  influence of genes and development. *Nature Neurosci.* 6:692–695.
- Zatorre, R. J., Belin, P. & Penhune, V. B. (2002). Structure and function
  of auditory cortex: music and speech. *Trends Cogn. Sci.* 6:37–46.
- Zatorre, R. J. et al. (1998). Functional anatomy of musical processing in
  listeners with absolute pitch and relative pitch. *PNAS* 95:3172–3177.
- Zeghidour, N. et al. (2021). LEAF: a learnable frontend for audio
  classification. *ICLR 2021*.
- Zhang, X. & Carney, L. H. (2005). Analysis of models for the synapse
  between the inner hair cell and the auditory nerve. *JASA* 118:1540–1553.
- Zilany, M. S. A., Bruce, I. C. & Carney, L. H. (2014). Updated parameters
  and expanded simulation options for a model of the auditory periphery.
  *JASA* 135:283–286 (building on Zilany et al., 2009, *JASA* 126:2390).

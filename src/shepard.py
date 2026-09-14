"""
Shepard-tone synthesis for the absolute-pitch (chroma/height) probe battery.

Biological grounding:
  * A Shepard tone (Shepard, 1964) contains only *octave-spaced* partials
    +/- a fixed "late" partial stack. Because every component is a power-of-two
    multiple of a common F0, the tone carries **pitch class** information but
    its **pitch height** (octave) is ambiguous: the spectral pattern is
    exactly invariant under octave translation.
  * A full tritone pair (Deutsch, 1986; tritone paradox) juxtaposes two such
    tones a half-octave (6 semitones) apart. Over a broad population the
    perceived *direction* (up vs down) splits depending on the pitch-class
    statistics of the listener's musical diet: the stimulus itself is
    directionally ambiguous.
  * `phase` is the key biophysical contrast we exploit:
      - "cos": all partials start in cosine phase -> sharp temporal waveform
        peaks -> strong **temporal (volley/periodicity) cues** (Rose et al.,
        1967; Cariani & Delgutte, 1996).
      - "rand": random partial phases -> flat, temporal-detail-free waveform
        -> only **place/rate (tonotopic) cues** survive (Oxenham, 2012;
        Shamma, 2001).
    Comparing chroma accuracy under the two phase conditions therefore guards
    the cochlear front-end's place-vs-temporal coding.

The partial lattice is anchored to A0 = 27.5 Hz so that pitch classes match
the 88-key MIDI/register indexing used in the piano experiments:
    f0(chroma c) = 27.5 * 2**(c/12),  partials at f0 * 2**n.
"""

import numpy as np


def shepard_partials(chroma, env_center_hz=800.0, f_lo=50.0, f_hi=6500.0,
                     half_width_oct=2.0):
    """Frequencies + raised-cosine spectral weights of a Shepard tone.

    Returns (freqs, env): the octave-spaced partial lattice centred on
    ``chroma`` and the log-frequency envelope weights (1 at the envelope
    peak, 0 an octave beyond ``half_width_oct``).

    ``env_center_hz`` is the same for every pitch class in a stimulus set:
    this is what makes the octave (height) ambiguous while preserving chroma.
    """
    f0 = 27.5 * (2.0 ** (chroma / 12.0))
    n_min = int(np.ceil(np.log2(f_lo / f0)))
    n_max = int(np.floor(np.log2(f_hi / f0)))
    if n_max < n_min:
        raise ValueError(f"no partials for chroma {chroma} in [{f_lo}, {f_hi}] Hz")
    ns = np.arange(n_min, n_max + 1)
    freqs = f0 * (2.0 ** ns)
    rel = np.log2(freqs / env_center_hz)
    env = np.where(np.abs(rel) <= half_width_oct,
                   0.5 * (1.0 + np.cos(np.pi * rel / half_width_oct)), 0.0)
    return freqs, env


def shepard_tone(chroma, env_center_hz=800.0, duration=0.1, fs=44100,
                 phase="cos", rng=None, **kw):
    """Synthesise one 100 ms Shepard tone for a pitch class (0..11).

    phase="cos"  -> all partials in cosine phase (strong temporal cues).
    phase="rand" -> random partial phases (place cues only).
    Returns (t, sig, freqs, env); peak-normalised to unit amplitude so the
    Poisson front-end sees loudness comparable to the training tones.
    """
    freqs, env = shepard_partials(chroma, env_center_hz=env_center_hz, **kw)
    t = np.arange(0.0, duration, 1.0 / fs)
    if phase == "cos":
        ph = np.zeros_like(freqs)
    elif phase == "rand":
        ph = (rng if rng is not None else np.random).uniform(
            0.0, 2.0 * np.pi, size=freqs.shape)
    else:
        raise ValueError("phase must be 'cos' or 'rand'")
    sig = np.sum(env[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :]
                                       + ph[:, None]), axis=0)
    peak = np.abs(sig).max()
    if peak > 0:
        sig = sig / peak
    return t, sig, freqs, env


def spectral_centroid_hz(freqs, env):
    """Amplitude-weighted spectral centroid (a place/rate height cue)."""
    w = env.sum()
    return float(np.sum(freqs * env) / w) if w > 0 else float("nan")
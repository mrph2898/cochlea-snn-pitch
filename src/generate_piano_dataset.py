"""
generate_piano_dataset.py
Creates a spike-train dataset covering the full 88-key piano range (A0–C8),
using the spikify filterbank + Poisson encoder.
Saves tensors to data/piano_dataset.pt for later training.
"""

import numpy as np
import torch
from spikify.filters import FilterBank
from spikify.encoders.rate import poisson

import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.preprocess import generate_tone, add_white_noise, gammatone_filterbank, ihc_rectification_compression, generate_poisson_spikes


# ---------- Piano note definitions ----------
# Standard 88-key piano: A0 (27.5 Hz) to C8 (4186.01 Hz)
# We will generate all 12 semitones per octave for completeness.
def piano_keys():
    """Returns list of (note_name, frequency) for 88 keys A0..C8."""
    A0 = 27.5
    notes = []
    for k in range(88):
        f = A0 * (2 ** (k / 12))
        # Note name (optional, could be generated more nicely)
        notes.append(f)
    return notes

def piano_key_labels():
    """Return list of 88 note names from A0 to C8."""
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    labels = []
    octave = 0
    # A0 is the 1st key (index 0 of our 88 keys)
    # In the 12‑note cycle, A is at index 0 if we start from A.
    for i in range(88):
        # note index wraps every 12 semitones
        idx = i % 12
        note = note_names[idx]
        # The octave changes at C (index 3): after B (index 2) comes C of the next octave.
        # Since we started at A0, the first C is C1 (i = 3).
        if idx == 3:   # C
            octave += 1
        labels.append(f"{note}{octave}")
    return labels

# If you strictly want the 7-note C major scale from the table, uncomment:
# PIANO_PITCHES = [
#     16.35, 18.35, 20.60, 21.83, 24.50, 27.50, 30.87,
#     32.70, 36.71, 41.20, 43.65, 49.00, 55.00, 61.74,
#     65.41, 73.42, 82.41, 87.31, 98.00, 110.00, 123.47,
#     130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
#     261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
#     523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,
#     1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53,
#     2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07,
#     4186.01, 4698.63, 5274.04, 5587.65, 6271.93, 7040.00, 7902.13
# ]

# ---------- Parameters ----------
FS = 44100
DURATION = 0.1  # 100 ms tone bursts
NUM_CHANNELS = 50      # filterbank channels
F_LOW = 100            # Hz – lower bound (will be adjusted for full range)
F_HIGH = 8000          # Hz – upper bound (extended to include high notes)
BIN_SIZE = int(FS / 1000)  # 1 ms bins (44.1 samples)
SAMPLES_PER_CLASS = 50
NOISE_VARIANCE = True   # add mild noise during generation for robustness

# ---------- Front-end ----------
def generate_spike_tensor(freq, filterbank, version='spikify', snr_db=None):
    """Returns binned spike tensor of shape (time_steps, num_channels)."""
    t, sig = generate_tone(freq, DURATION, FS)
    if snr_db is not None:
        sig = add_white_noise(sig, snr_db)

    if version == 'spikify':
        # Decompose and encode
        filtered = filterbank.decompose(sig)  # (timesteps, channels, features)
        filtered = filtered.reshape(-1, filtered.shape[1] * filtered.shape[2])
        spikes_44k = poisson(filtered, interval_length=2).T  # (channels, timesteps)
    elif version == 'handy':
        filtered, _ = gammatone_filterbank(sig, FS, num_channels=NUM_CHANNELS)
        ihc = ihc_rectification_compression(filtered)
        # Use your new rel_threshold logic here
        spikes_44k = generate_poisson_spikes(ihc, FS, max_rate=800)
    else:
        raise NotImplementedError("Only spikify version is used here.")

    # Temporal binning to 1 ms steps
    num_bins = spikes_44k.shape[1] // BIN_SIZE
    binned = np.zeros((NUM_CHANNELS, num_bins))
    for i in range(num_bins):
        window = spikes_44k[:, i*BIN_SIZE : (i+1)*BIN_SIZE]
        binned[:, i] = (window.sum(axis=1) > 0).astype(float)

    return torch.tensor(binned, dtype=torch.float32).T  # (time, channels)

# ---------- Dataset creation and saving ----------
def create_and_save_piano_dataset(
    version='spikify',
    samples_per_class=50,
    noise_variance=True,
    save_path=None,
):
    """
    Generates spike tensors for all piano keys (88 keys) and saves them.
    Adjusts f_low/f_high if necessary to cover the full range.
    """
    if save_path is None:
        save_path = str(ROOT / "data" / "piano_dataset")
    # If the requested frequency range doesn't cover the piano, warn and adjust
    keys = piano_keys()   # 88 frequencies A0..C8
    min_f, max_f = min(keys), max(keys)
    actual_f_low = min(F_LOW, min_f)
    actual_f_high = max(F_HIGH, max_f)

    if actual_f_low != F_LOW or actual_f_high != F_HIGH:
        print(f"Piano range [{min_f:.1f}, {max_f:.1f}] Hz extends beyond filterbank limits.")
        print(f"Expanding filterbank from [{F_LOW}, {F_HIGH}] → [{actual_f_low}, {actual_f_high}].")

    # Create filterbank once to avoid re-initialising for every tone
    filterbank = FilterBank(
        fs=FS,
        channels=NUM_CHANNELS,
        f_min=actual_f_low,
        f_max=actual_f_high,
        filter_type='gammatone',
        order=4
    )

    X_list, y_list = [], []
    for label, freq in enumerate(keys):
        print(f"Generating {samples_per_class} samples for {freq:.2f} Hz ({label+1}/88)...")
        for _ in range(samples_per_class):
            snr = np.random.uniform(10, 30) if noise_variance else None
            spike_tensor = generate_spike_tensor(freq, filterbank, version=version, snr_db=snr)
            X_list.append(spike_tensor)
            y_list.append(label)

    X = torch.stack(X_list)  # (N, time, channels)
    y = torch.tensor(y_list, dtype=torch.long)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'X': X, 'y': y, 'labels': keys}, save_path + f"_{version}.pt")
    print(f"\nDataset saved to {save_path+ f"_{version}.pt"}")
    print(f"Shape: X = {X.shape}, y = {y.shape}")

if __name__ == "__main__":
    create_and_save_piano_dataset(version="handy")
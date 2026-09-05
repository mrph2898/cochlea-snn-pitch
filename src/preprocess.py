"""
Auditory preprocessing pipeline for SNN Pitch Perception.
Translates audio waveforms into multi-channel spike trains.
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
from spikify.filters import FilterBank
from spikify.encoders.rate import poisson

ROOT = Path(__file__).resolve().parent.parent


def generate_tone(freq, duration, fs=44100, phase=0):
    """Generates a pure sine wave tone."""
    t = np.arange(0, duration, 1/fs)
    return t, np.sin(2 * np.pi * freq * t + phase)

def add_white_noise(sig, snr_db):
    """Adds Gaussian white noise to a signal at a specified SNR in dB."""
    sig_power = np.mean(sig ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(sig))
    return sig + noise

class LinearERB:
    """
    Standard ERB-rate scale (Glasberg & Moore, 1990).
    ERB(F) = 24.7 * (4.37*F + 1) [Hz]. F in kHz, 
    scale ERBS(f) = 21.4 * log_{10}(1 + 4.37*F) [kHz]
    """
    
    def __init__(self):
        self._c = 1000 / (24.7 * 4.37)   # constant for the scale
    
    def __call__(self, f):
        """f in Hz -> ERB-rate number."""
        # (Uncomment for bounds checking, if needed)
        # assert 100 < f < 6500, "Frequency must be 0.1–6.5 kHz in Hz"
        return self._c * np.log(4.37 * f / 1000.0 + 1)
    
    def erb(self, f):
        return (1 / self._c * f + 24.7) * 1000
    
    def inv(self, e):
        """ERB-rate number -> frequency in Hz."""
        return (np.exp(e / self._c) - 1) / 4.37 #* 1000.0
    
class QuadERB:
    """
    Quadratic ERB-rate scale.
    ERB(F) = 6.23*F^2 + 93.39*F + 28.52 [Hz], F is a centered freq in kHz
    and the scale equals to ERBS(F) = 11.17*ln((F + 0.312) / (F + 14.675)) + 43.0, F in kHz
    Valid for f \\in [0.1, 6.5] kHz
    """
    
    def __init__(self):
        self._A = 6.23
        self._B = 93.39
        self._C = 28.52
        R = np.sqrt(self._B**2 - 4 * self._A * self._C)
        self._R = R
        self._C0 = (1 / R) * np.log(np.abs((self._B - R) / (self._B + R)))
        self._f1 = (self._B - R) / (2 * self._A)
        self._f2 = (self._B + R) / (2 * self._A)
    
    def __call__(self, f):
        """f in Hz -> ERB-rate number in Hz."""
        # assert 100 < f < 6500
        f_khz = f / 1000.0
        return ((1 / self._R) * np.log(
            np.abs((f_khz + self._f1) / (f_khz + self._f2))) - self._C0) * 1000

    def erb(self, f):
        f_khz = f / 1000.0
        return (self._A * f_khz**2 + self._B * f_khz + self._C) # * 1000
    
    def inv(self, e):
        """ERB-rate number in Hz -> frequency in Hz."""
        exp_term = np.exp(self._R * (e / 1000 + self._C0))
        f_khz = (self._f1 - exp_term * self._f2) / (exp_term - 1)
        return f_khz * 1000.0
    
class VoiceboxERB:
    """Voicebox-style ERB-rate scale.
    (http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/frq2erb.html)
    ERBS(f) = a*ln(h - k / (f + c)) [Hz], f in Hz
        for p=-0.312, q=-14.7, a = 1000/(6.23*(p-q))= 11.17268,
            h = q/p = 47.065,  k = 1000(q - q^2/p) = 676170.42, 
            c = -1000q = 14678.49
    f = k / (h - exp(ERBS(f) / a)) - c
    """
    
    def __init__(self):
        self._p = -0.312
        self._q = -14.67849
        self._a = 1000 / (6.23 * (self._p - self._q))
        self._h = self._q / self._p
        self._k = 1000 * (self._q - self._q**2 / self._p)
        self._c = -1000 * self._q
        self._erb_class = QuadERB()
    
    def __call__(self, f):
        """f in Hz -> ERB-rate number."""
        # assert 100 < f < 6500
        return self._a * np.log(self._h - self._k / (f + self._c))
    
    def erb(self, f):
        return self._erb_class.erb(f)

    def inv(self, e):
        """ERB-rate number -> frequency in Hz."""
        return self._k / (self._h - np.exp(e / self._a)) - self._c
    

def erb_space(low_freq, high_freq, num_channels, approx="linear"):
    """
    Equivalent Rectangular Bandwidth scaling for on a smoother perceptual frequency axis.
    Calculates center frequencies evenly spaced on the ERB scale 
    by integrating the reciprocal of ERB(f) function approximation:
      - for 'quad':   
      - for 'linear': 
      - 'voicebox' 
    Parameters:
        low_freq (float): the lowest freq of the bandwidth in Hz 
        high_freq (float): the highest freq of the bandwidth in Hz 
        num_channels (int): the number of channels in the filter bank
    """
    assert low_freq > 20, "low_freq_khz must be > 20 Hz"
    assert low_freq < 20000, "low_freq_khz must be < 20000 Hz"
    assert high_freq > 20, "high_freq_khz must be > 20 Hz"
    assert high_freq < 20000, "high_freq_khz must be < 20000 Hz"
    assert low_freq < high_freq, "low_freq must be less than high_freq"

    match approx:
        case 'linear':
            erb = LinearERB()
        case "quad":
            erb = QuadERB()
        case "voicebox":
            erb = VoiceboxERB()
        case _:
            raise NotImplemented(f"Unsupported erb type {approx}")

    erb_spacing = np.linspace(erb(low_freq), erb(high_freq), num_channels)
    center_freqs = erb.inv(erb_spacing)
    return center_freqs, erb

def gammatone_filterbank(sig, fs, 
                         num_channels=50, f_low=100, f_high=6500,
                         erb_approx='voicebox'):
    """
    Applies a time-domain FIR Gammatone filterbank.
    Equation: h_i(t) = t^{n-1} * e^{-2*pi*b_i*t} * cos(2*pi*f_i*t + phi)
    """
    cfs, erbs_class = erb_space(f_low, f_high, num_channels, approx=erb_approx)
    filtered_signals = np.zeros((num_channels, len(sig)))
    
    # Filter length: 50ms is usually sufficient for impulse response to decay
    t_ir = np.arange(0, 0.05, 1/fs) 
    order = 4
    
    for i, fc in enumerate(cfs):
        # Calculate bandwidth based on ERB
        erb = erbs_class.erb(fc)
        b = 1.019 * erb
        
        # Gammatone impulse response
        h = (t_ir**(order-1)) * np.exp(-2 * np.pi * b * t_ir) * np.cos(2 * np.pi * fc * t_ir)
        # Normalize the impulse response
        h = h / np.max(np.abs(h))
        
        # Apply filter via FFT convolution for speed
        filtered = signal.fftconvolve(sig, h, mode='same')
        filtered_signals[i, :] = filtered
        
    return filtered_signals, cfs

def ihc_rectification_compression(filtered_signals, exponent=0.7):
    """
    Simulates Inner Hair Cell (IHC) dynamics:
    Soft half-wave rectification followed by power-law compression.
    """
    # Half-wave rectification
    rectified = np.maximum(0, filtered_signals)
    # Power-law compression
    compressed = rectified ** exponent
    return compressed

# def generate_poisson_spikes(rate_signals, fs, max_rate=500):
#     """
#     Generates spikes using an inhomogeneous Poisson process.
#     Rate is proportional to the IHC output.
#     """
#     dt = 1 / fs
#     num_channels, time_steps = rate_signals.shape
#     spikes = np.zeros_like(rate_signals, dtype=int)
    
#     for i in range(num_channels):
#         channel_rate = rate_signals[i]
#         # Normalize to max_rate (Hz) to keep spike counts biologically plausible
#         if np.max(channel_rate) > 0:
#             channel_rate = (channel_rate / np.max(channel_rate)) * max_rate
            
#         prob_of_spike = channel_rate * dt
#         # Generate spikes where random uniform value is less than probability
#         spikes[i] = (np.random.uniform(0, 1, time_steps) < prob_of_spike).astype(int)
        
#     return spikes

def generate_poisson_spikes(rate_signals, fs, max_rate=500):
    """
    Generates spikes using an inhomogeneous Poisson process.
    Rate is proportional to the IHC output.
    """
    dt = 1 / fs
    num_channels, time_steps = rate_signals.shape
    spikes = np.zeros_like(rate_signals, dtype=int)
    
    # 1. Find the global maximum across ALL channels to preserve relative loudness
    global_max = np.max(rate_signals)
    
    for i in range(num_channels):
        channel_rate = rate_signals[i]
        
        # 2. Scale relative to the global loudest point
        if global_max > 0:
            # High-energy channels will be near max_rate
            # Low-energy channels will stay near 0 Hz
            scaled_rate = (channel_rate / global_max) * max_rate
        else:
            scaled_rate = channel_rate
            
        prob_of_spike = scaled_rate * dt
        spikes[i] = (np.random.uniform(0, 1, time_steps) < prob_of_spike).astype(int)
        
    return spikes

def plot_raster(spikes, fs, title="Auditory Nerve Spike Raster"):
    """Plots a raster plot of the spike trains."""
    num_channels, time_steps = spikes.shape
    time_axis = np.arange(time_steps) / fs
    
    spike_times = []
    spike_channels = []
    
    for i in range(num_channels):
        times = time_axis[spikes[i] == 1]
        spike_times.extend(times)
        spike_channels.extend([i] * len(times))
        
    plt.figure(figsize=(10, 6))
    plt.scatter(spike_times, spike_channels, s=2.5, color='black', marker='|')
    plt.xlabel("Time (s)")
    plt.ylabel("Cochlear Channel Index")
    plt.title(title)
    plt.ylim(-1, num_channels)
    plt.tight_layout()
    plt.show()

def plot_combined_results(t, clean_sig, noisy_sig,
                          spikes_clean, spikes_noisy,
                          tone_freq,
                          xlim=None,
                          savefig=None
                          ):
    """
    Visualizes signals and rasters with frequency-mapped Y-axes.
    """
    num_channels = spikes_clean.shape[0]

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    # --- Subplot 1: Waveforms ---
    ax[0].plot(t, clean_sig, color='tab:blue', lw=1.5, label=f'Clean Tone ({tone_freq} Hz)', alpha=0.8)
    ax[0].plot(t, noisy_sig, color='tab:red', lw=1.0, alpha=0.4, label='Noisy Tone')
    ax[0].set_ylabel('Amplitude', fontsize=14)
    ax[0].legend(loc='upper right')
    ax[0].set_title('Synchronized Cochlear Response', fontsize=16, fontweight='bold')

    # --- Subplot 2 & 3: Rasters ---
    for idx, (spike_data, label, color) in enumerate([
        (spikes_clean, 'Clean CF (Hz)', 'black'),
        (spikes_noisy, 'Noisy CF (Hz)', 'firebrick')
    ]):
        current_ax = ax[idx + 1]
        for i in range(num_channels):
            if spike_data[i, :].sum() > 0:
                t_sp = t[spike_data[i, :] > 0.5]
                # Use cfs[i] instead of (i+1) for the y-position
                current_ax.plot(t_sp, np.ones(len(t_sp)) * cfs[i], 
                                '|', color=color, ms=8, markeredgewidth=1.5)
        
        current_ax.set_ylabel(label, fontsize=14)
        # Use log scale if your frequencies cover a wide range (e.g., 100Hz - 6kHz)
        current_ax.set_yscale('log') 
        current_ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        
        # Set ticks to show meaningful frequencies
        tick_indices = np.linspace(0, num_channels - 1, 6, dtype=int)
        current_ax.set_yticks(cfs[tick_indices])

    ax[2].set_xlabel('Time (s)', fontsize=14)

    if xlim:
        ax[0].set_xlim(xlim)
    
    plt.tight_layout()
    if savefig:
        out_dir = ROOT / "data" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'filterbank_freq{tone_freq}_{savefig}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_erbs():
    # Frequency axes
    f_rate = np.logspace(np.log10(20), np.log10(20000), 1000)   # for ERB‑rate
    f_bw = np.logspace(np.log10(20), np.log10(20000), 1000)     # for bandwidth

    # Compute values
    erb_linear = LinearERB()
    erb_quad   = QuadERB()
    erb_vb     = VoiceboxERB()

    rate_lin = erb_linear(f_rate)
    rate_quad = erb_quad(f_rate)
    rate_vb = erb_vb(f_rate)

    bw_lin = erb_linear.erb(f_bw)
    bw_quad = erb_quad.erb(f_bw)   # Voicebox is identical, so we don't plot it separately

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

    # ---- Subplot 1: ERB‑rate scales ----
    ax1.semilogx(f_rate, rate_lin, label='Linear (Standard)',
                 linestyle='-', linewidth=2)
    ax1.semilogx(f_rate, rate_quad, label='Quadratic',
                 linestyle='--', linewidth=2)
    ax1.semilogx(f_rate, rate_vb, label='Voicebox',
                 linestyle='-.', linewidth=2)
    ax1.set_ylabel('ERB-rate (number)', fontsize=12)
    ax1.set_title('Comparison of ERB-rate scales', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, which='both', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)

    # ---- Subplot 2: ERB bandwidths + valid ranges ----
    ax2.semilogx(f_bw, bw_lin, label='Linear (Standard)',
                 color='C0', linewidth=2)
    ax2.semilogx(f_bw, bw_quad*1000, label='Quadratic / Voicebox',
                 color='C1', linestyle='--', linewidth=2)

    # Shade valid ranges
    ax2.axvspan(100, 10000, alpha=0.15, color='C0',
                label='Linear valid range (100 Hz - 10 kHz)')
    ax2.axvspan(100, 6500, alpha=0.1, color='C1',
                label='Quad valid range (100 Hz - 6.5 kHz)')

    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('ERB bandwidth (Hz)', fontsize=12)
    ax2.set_title('ERB bandwidths with valid‑range spans', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, which='both', linestyle=':', alpha=0.7)
    ax2.set_xlim(20, 20000)

    plt.tight_layout()
    out_dir = ROOT / "data" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'erb_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def test_erbs():
    f_test = np.logspace(np.log10(100), np.log10(6500), 1000)  # 100 Hz → 6.5 kHz
    tolerance = 1e-6  # Hz – more than enough for 64‑bit float

    models = {
        'Linear':   LinearERB(),
        'Quadratic': QuadERB(),
        'Voicebox':  VoiceboxERB()
    }

    all_passed = True
    for name, erb in models.items():
        f_recovered = erb.inv(erb(f_test))
        error = np.abs(f_recovered - f_test)
        max_err = np.max(error)
        passed = max_err < tolerance
        all_passed &= passed

        print(f"{name:10s}  max round‑trip error = {max_err:.2e} Hz  "
            f"({'PASS' if passed else 'FAIL'})")

    # Optionally, a hard assertion to crash the script if any model fails
    assert all_passed, "One or more ERB models failed the round‑trip test!"
    print("\nAll models passed: inv(call(f)) == f within numerical precision.")


if __name__ == "__main__":
    # --- Experiment Setup ---
    fs = 44100
    num_channels = 50
    f_low = 100 
    f_high = 6500
    duration = 0.1  # 100 ms burst
    # freq_C4 = 261.63 # C4 tone
    freq_A4 = 880. #261.63
    tone_freq = freq_A4

    
    # 0. Plot comparing erbs
    # test_erbs()
    plot_erbs()
    # 1. Clean Tone Processing
    print(f"Processing Clean {tone_freq} Hz tone...")
    t, clean_sig = generate_tone(tone_freq, duration, fs)
    filtered_clean, cfs = gammatone_filterbank(
        clean_sig, fs, num_channels=num_channels,
        f_low=f_low, f_high=f_high
    )
    ihc_clean = ihc_rectification_compression(filtered_clean)
    spikes_clean = generate_poisson_spikes(ihc_clean, fs, max_rate=800)
    
    # Plot clean raster (Notice the phase locking / volley patterns!)
    # plot_raster(spikes_clean, fs, title="Clean C4 Tone (261.63 Hz) - Poisson Spikes")

    # 2. Noisy Tone Processing (0 dB SNR)
    print("Processing Noisy tone (0 dB SNR)...")
    noisy_sig = add_white_noise(clean_sig, snr_db=0)
    filtered_noisy, _ = gammatone_filterbank(
        noisy_sig, fs, num_channels=num_channels,
        f_low=f_low, f_high=f_high
    )
    ihc_noisy = ihc_rectification_compression(filtered_noisy)
    spikes_noisy = generate_poisson_spikes(ihc_noisy, fs, max_rate=800)
    
    # Plot noisy raster
    # plot_raster(spikes_noisy, fs, title="Noisy C4 Tone (0 dB SNR) - Poisson Spikes").
    # Zooming in on a 30ms window to clearly see the phase-locking
    plot_combined_results(t, clean_sig, noisy_sig, 
                        spikes_clean, spikes_noisy,
                        tone_freq=tone_freq, 
                        xlim=[0.03, 0.06], 
                        savefig="handy_fixed"
                        )
    
    # Compare with the spikify
    filter = FilterBank(fs=fs, channels=num_channels,
                        f_min=f_low, f_max=f_high,
                        filter_type='gammatone', order=4
                        )

    filtered_signal = filter.decompose(clean_sig) # (timesteps, channels, features)

    filtered_signal = np.reshape(filtered_signal, (-1, filtered_signal.shape[1] * filtered_signal.shape[2]))

    # Encode the filtered signal
    encoded_signal = poisson(filtered_signal, interval_length=2).T

    filtered_signal_noisy = filter.decompose(noisy_sig) # (timesteps, channels, features)

    filtered_signal_noisy = np.reshape(filtered_signal_noisy, (-1, filtered_signal_noisy.shape[1] * filtered_signal_noisy.shape[2]))

    # Encode the filtered signal
    encoded_signal_noisy = poisson(filtered_signal_noisy, interval_length=2).T
    plot_combined_results(t, clean_sig, noisy_sig, 
                        encoded_signal, encoded_signal_noisy,
                        tone_freq=tone_freq, 
                        xlim=[0.03, 0.06], 
                        savefig="library"
                        )


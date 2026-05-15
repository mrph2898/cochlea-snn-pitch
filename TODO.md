# Project Roadmap & Next Steps

## Week 3: SNN Core Setup (`src/network.py`)
- [ ] Initialize an `snnTorch` model.
- [ ] Create a Feed-Forward architecture: 50 inputs (from `preprocess.py`) -> 100 Excitatory Leaky Integrate-and-Fire (LIF) neurons -> 3 Output neurons.
- [ ] Configure LIF parameters to match biological plausibility (e.g., tau_m ~ 20 ms, V_th).

## Week 4: Learning Rule (`src/learning.py`)
- [ ] Implement Unsupervised Spike-Timing-Dependent Plasticity (STDP). 
      *Note: Since `snnTorch` relies primarily on surrogate gradient descent, you will need to either write a custom PyTorch autograd function for pair-based STDP (LTP for pre-before-post, LTD for post-before-pre), or adapt a local learning rule.*
- [ ] Write a training loop in `experiments/run_stdp_training.py` presenting C4, E4, and G4 tones (100 trials each).
- [ ] Plot synaptic weight convergence.

## Week 5: Comparative Baseline (`src/baseline.py` & `experiments/noise_robustness.py`)
- [ ] Build a rate-based Multi-Layer Perceptron (MLP) in standard PyTorch using the exact same layer dimensions.
- [ ] Train the MLP on continuous firing rates (float values) instead of spikes.
- [ ] Write the evaluation script: loop through SNRs (e.g., -10dB to +20dB), test both models, and plot `Classification Accuracy vs. SNR`.
- [ ] Validate the hypothesis: The SNN should exhibit a flatter degradation slope due to temporal coincidence detection.

## Week 6: Reporting
- [ ] Compile workflows into Jupyter Notebooks (`notebooks/`) for easy visualization and grading.
- [ ] Extract the final SNR figure for the presentation.
# Cochlea-Inspired Spiking Neural Network for Robust Pitch Perception

This repository explores the noise-robustness of temporal coding in auditory perception using Spiking Neural Networks (SNNs). It implements a biological auditory front-end (Gammatone + IHC) and routes the resulting spike trains into a PyTorch-based SNN using `snnTorch`.

## Installation

This project uses `uv` for lightning-fast Python package management. 

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
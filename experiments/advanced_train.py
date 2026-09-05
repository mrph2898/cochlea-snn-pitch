import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import scipy.signal as signal

import snntorch as snn
from snntorch import surrogate

# Import spikify components
# Note: Spikify typically provides delta modulators or threshold crossing encoders
from spikify.encoders import LatencyEncoder, DeltaModulator 

# --- Hyperparameters ---
FS = 16000  # Lowering sampling rate to 16kHz simplifies temporal sequence processing
DURATION = 0.1  # 100 ms
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PITCHES = [261.63, 329.63, 392.00]  # C4, E4, G4

# --- 1. Audio Generation Functions ---
def generate_tone(freq, duration, fs=16000):
    t = np.arange(0, duration, 1/fs)
    sig = np.sin(2 * np.pi * freq * t)
    window = signal.windows.tukey(len(sig), alpha=0.1)
    return t, sig * window

def add_white_noise(sig, snr_db):
    sig_power = np.mean(sig ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(sig))
    return sig + noise

# --- 2. Advanced SNN Network with spikify Compatibility ---
class AdvancedPitchSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden=128, num_outputs=3, beta=0.85):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # We process input channels over time
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        # Membrane logit output: high threshold, no reset
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1e9)

    def forward(self, x):
        """
        x shape: (Time, Batch, Channels)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        time_steps = x.size(0)
        
        # Iterate over time
        for step in range(time_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)
            
        # Instead of taking the mean over messy histories, we return the final state 
        # of the membrane potential accumulator after reading the whole audio frame.
        return mem2 

# --- 3. Dataset Generation via spikify ---
def create_spikify_dataset(samples_per_class=40):
    X, y = [], []
    
    # Initialize Delta Modulator from spikify
    # It tracks changes in amplitude and emits up/down spikes
    encoder = DeltaModulator(threshold=0.01, descending_threshold=0.01)
    
    print("Encoding audio signals into spike trains using spikify...")
    for label, freq in enumerate(PITCHES):
        for _ in range(samples_per_class):
            _, sig = generate_tone(freq, DURATION, FS)
            
            # Inject varying train noise to enforce feature learning
            snr = np.random.uniform(5, 25)
            sig = add_white_noise(sig, snr)
            
            # Convert to PyTorch float tensor for spikify
            sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0) # (1, Samples)
            
            # spikify encoders return structural spike formats (typically positive and negative changes)
            spikes = encoder(sig_tensor) 
            
            # Flatten or handle channels from spikify output
            # Usually spikes shape is (Batch, Channels, Time) or similar depending on variant
            spikes = spikes.squeeze(0).float() # (Channels, Time)
            
            # Reshape to snnTorch expectations: (Time, Channels)
            spikes = spikes.transpose(0, 1)
            
            X.append(spikes)
            y.append(label)
            
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y), X.shape[-1] # Return datasets and the channel size

# --- 4. Training Loop ---
def train_network(model, dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining Model on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.transpose(0, 1).to(DEVICE) # (Time, Batch, Channels)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            final_mem = model(X_batch)
            
            loss = criterion(final_mem, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = final_mem.max(1)
            correct += (predicted == y_batch).sum().item()
            
        acc = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc*100:.2f}%")

# --- 5. Robustness Evaluation Loop ---
def evaluate_robustness(model, num_channels):
    model.eval()
    encoder = DeltaModulator(threshold=0.01, descending_threshold=0.01)
    snr_levels = [20, 10, 5, 0, -5, -10]
    accuracies = []
    
    print("\n--- Evaluating spikify Front-End Noise Robustness ---")
    for snr in snr_levels:
        correct = 0
        trials = 30
        
        for _ in range(trials):
            for label, freq in enumerate(PITCHES):
                _, sig = generate_tone(freq, DURATION, FS)
                sig = add_white_noise(sig, snr)
                
                sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    spikes = encoder(sig_tensor).squeeze(0).float().transpose(0, 1)
                    X = spikes.unsqueeze(1).to(DEVICE) # Add Batch Dim -> (Time, 1, Channels)
                    
                    final_mem = model(X)
                    _, predicted = final_mem.max(1)
                    if predicted.item() == label:
                        correct += 1
                        
        acc = correct / (trials * len(PITCHES))
        accuracies.append(acc * 100)
        print(f"SNR: {snr:3d} dB | Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    # Generate unified structured spikes via spikify
    dataset, num_features = create_spikify_dataset(samples_per_class=70)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Instantiate architecture
    model = AdvancedPitchSNN(num_inputs=num_features).to(DEVICE)
    
    # Execute Pipelines
    train_network(model, train_loader)
    evaluate_robustness(model, num_features)
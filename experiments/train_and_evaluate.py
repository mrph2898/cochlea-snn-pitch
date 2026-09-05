"""
End-to-end training and noise robustness evaluation.
Combines Week 3, 4, and 5 objectives.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # optional – makes heatmaps prettier

# Import your front-end and network
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.preprocess import generate_tone, add_white_noise, gammatone_filterbank, ihc_rectification_compression, generate_poisson_spikes
from src.snn import PitchSNN
import src.generate_piano_dataset as gpd

from spikify.filters import FilterBank
from spikify.encoders.rate import poisson

# --- Hyperparameters ---
FS = 44100
DURATION = 0.1  # 100 ms
NUM_CHANNELS = 50
BATCH_SIZE = 16
EPOCHS = 20
F_LOW = 100
F_HIGH = 6500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PITCHES = [261.63, 440.00, 880.00]  # C4, A4, A5

# Artifact directories (root-relative, safe from any CWD)
MODEL_DIR = ROOT / "data" / "trained_models"
PLOT_DIR = ROOT / "data" / "plots"
DATASET_DIR = ROOT / "data"


def plot_confusion(y_true, y_pred, classes, title="Confusion Matrix",
                   save_path=None, label_names=None):
    """
    If label_names is provided (list of 88 strings), uses them as axis tick labels.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(classes))
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                     xticklabels=label_names if label_names else range(classes),
                     yticklabels=label_names if label_names else range(classes))
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Optional: colour black‑key labels dark grey, white‑key labels black
    if label_names is not None:
        for label in ax.get_xticklabels():
            note = label.get_text()
            if '#' in note:   # black key
                label.set_color('darkgrey')
        for label in ax.get_yticklabels():
            note = label.get_text()
            if '#' in note:
                label.set_color('darkgrey')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def generate_spike_tensor(freq, filterbank=None, version='handy', snr_db=None):
    """Passes a tone through the biological front-end and returns binned spikes."""
    t, sig = generate_tone(freq, DURATION, FS)
    if snr_db is not None:
        sig = add_white_noise(sig, snr_db)

    if version == 'handy':
        filtered, _ = gammatone_filterbank(sig, FS, num_channels=NUM_CHANNELS)
        ihc = ihc_rectification_compression(filtered)
        # Use your new rel_threshold logic here
        spikes_44k = generate_poisson_spikes(ihc, FS, max_rate=800)
    elif version == 'spikify':
        filtered_signal = filterbank.decompose(sig) # (timesteps, channels, features)

        filtered_signal = np.reshape(filtered_signal, (-1, filtered_signal.shape[1] * filtered_signal.shape[2]))

        # Encode the filtered signal
        # shape (num_channels, timesteps)
        spikes_44k = poisson(filtered_signal, interval_length=2).T

    else:
        raise NotImplementedError("Choose the proper version of spike generator: handy or spikify")
    
    # --- TEMPORAL BINNING ---
    # Bin 44.1kHz spikes into 1kHz (1ms) windows to make SNN training fast
    # 44100 Hz * 0.1s = 4410 steps -> down to 100 steps
    bin_size = int(FS / 1000) 
    num_bins = spikes_44k.shape[1] // bin_size
    
    binned_spikes = np.zeros((NUM_CHANNELS, num_bins))
    for i in range(num_bins):
        window = spikes_44k[:, i*bin_size : (i+1)*bin_size]
        # If there's at least one spike in the 1ms window, mark as 1
        binned_spikes[:, i] = (np.sum(window, axis=1) > 0).astype(float)
        
    # snnTorch expects shape: (Time, Batch, Channels). We return (Time, Channels) here.
    return torch.tensor(binned_spikes, dtype=torch.float32).transpose(0, 1)

def create_dataset(samples_per_class=50, filterbank=None, noise_variance=True, version='handy'):
    """Creates a dataset of spikes. Adds random slight noise to prevent overfitting."""
    X, y = [], []
    for label, freq in enumerate(PITCHES):
        for _ in range(samples_per_class):
            # Train with clean and slightly noisy data for robustness
            snr = np.random.uniform(10, 30) if noise_variance else None
            spike_tensor = generate_spike_tensor(freq, snr_db=snr, filterbank=filterbank, version=version)
            X.append(spike_tensor)
            y.append(label)
    
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y)

def train_network(model, dataloader, version="handy"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []

    model.train()
    print(f"Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.transpose(0, 1).to(DEVICE) # -> (Time, Batch, Channels)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass returns membrane potential history: (Time, Batch, Classes)
            mem_out = model(X_batch)
            
            # Calculate logits by averaging membrane potential over the time domain
            logits = mem_out.mean(dim=0) 
            
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / len(dataloader.dataset)
        
        # ----- ADD -----
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
            
        acc = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc*100:.2f}%")

     # ----- SAVE MODEL & PLOTS (AFTER TRAINING) -----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f'model_supervised_{version}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot loss & accuracy
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, EPOCHS+1), train_losses, color='tab:red', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(range(1, EPOCHS+1), [a*100 for a in train_accs], color='tab:blue', marker='s')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Training Progress (Supervised)')
    fig.tight_layout()
    plt.savefig(PLOT_DIR / f'training_curves_supervised_{version}.pdf')
    plt.show()

    return model


def evaluate_noise_robustness(model, filterbank=None, version="handy"):
    """Tests the model at progressively worse Signal-to-Noise Ratios."""
    model.eval()
    snr_levels = [20, 10, 5, 0, -5, -10]
    accuracies = []
    
    print("\n--- Evaluating Noise Robustness ---")
    for snr in snr_levels:
        correct = 0
        trials = 30
        for _ in range(trials):
            for label, freq in enumerate(PITCHES):
                X = generate_spike_tensor(freq, snr_db=snr, filterbank=filterbank, version=version).unsqueeze(1).to(DEVICE) # Add batch dim
                y = torch.tensor([label]).to(DEVICE)
                
                # Inside evaluation loop...
                with torch.no_grad():
                    mem_out = model(X)
                    logits = mem_out.mean(dim=0)
                    _, predicted = logits.max(1)
                    if predicted == y:
                        correct += 1
                        
        acc = correct / (trials * len(PITCHES))
        accuracies.append(acc * 100)
        print(f"SNR: {snr:3d} dB | Accuracy: {acc*100:.2f}%")
        
    # Plotting the result
    plt.figure(figsize=(8, 5))
    plt.plot(snr_levels, accuracies, marker='o', color='darkblue', lw=2)
    plt.axhline(100/3, color='red', linestyle='--', label='Random Guessing (33%)')
    plt.gca().invert_xaxis()  # Reverse x-axis so noise increases to the right
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=14)
    plt.ylabel('Classification Accuracy (%)', fontsize=14)
    plt.title('SNN Noise Robustness (Temporal Coding)', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'noise_robustness_{version}.png')
    plt.show()


def train_on_piano_dataset(
    dataset_path=None,
    version="spikify",
    batch_size=64,
    epochs=20,
    learning_rate=0.005,
    device=DEVICE
):
    """
    Loads the pre‑generated piano dataset and trains PitchSNN on 88 classes.
    
    The model is re‑instantiated with num_classes=88 to match the full piano range.
    Training progress and final test accuracy are printed.
    """
    if dataset_path is None:
        dataset_path = str(DATASET_DIR / f'piano_dataset_{version}.pt')
    elif not os.path.isabs(dataset_path):
        dataset_path = str(DATASET_DIR / os.path.basename(dataset_path))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run generate_piano_dataset.py first.")
    
    data = torch.load(dataset_path, map_location='cpu')
    X, y = data['X'], data['y']          # X: (N, time, channels), y: (N,)
    num_classes = len(torch.unique(y))   # should be 88
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    # Use a simple split for a final test set (optional)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Initialize model for 88 classes ---
    model = PitchSNN(num_outputs=num_classes).to(device)
    
    # --- Training setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {device} with {num_classes} classes...")
    model.train()
    
    train_losses = []
    train_accs = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            # Input shape: (batch, time, channels) → (time, batch, channels)
            X_batch = X_batch.transpose(0, 1).to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            mem_out = model(X_batch)                  # (time, batch, classes)
            logits = mem_out.mean(dim=0)              # average over time
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == y_batch).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / len(train_loader.dataset)
        
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
            
        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc*100:.2f}%")

     # ----- SAVE MODEL & PLOTS (AFTER TRAINING) -----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f'model_piano88_{version}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot loss & accuracy
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, EPOCHS+1), train_losses, color='tab:red', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(range(1, EPOCHS+1), [a*100 for a in train_accs], color='tab:blue', marker='s')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Training Progress (Supervised)')
    fig.tight_layout()
    plt.savefig(PLOT_DIR / f'training_curves_piano88_{version}.pdf')
    plt.show()

    
    # --- Final evaluation on test set ---
    model.eval()
    correct_test = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.transpose(0, 1).to(device)
            y_batch = y_batch.to(device)
            mem_out = model(X_batch)
            logits = mem_out.mean(dim=0)
            _, predicted = logits.max(1)
            correct_test += (predicted == y_batch).sum().item()
            all_preds.append(predicted.cpu())
            all_labels.append(y_batch.cpu())
    test_acc = correct_test / len(test_loader.dataset)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    key_names = gpd.piano_key_labels()
    plot_confusion(all_labels, all_preds, classes=88,
                   title='Piano Pitch Confusion Matrix',
                   save_path=f'confusion_piano88_{version}.pdf',
                   label_names=key_names)
    
    return model


def main(train_regime="3classes", version='spikify'):
    match train_regime:
        case "3classes":
            filterbank = FilterBank(fs=FS, channels=NUM_CHANNELS,
                                f_min=F_LOW, f_max=F_HIGH,
                                filter_type='gammatone', order=4
                                )
            # 1. Prepare Data
            print("Generating training dataset (this may take a minute)...")
            train_dataset = create_dataset(samples_per_class=60, filterbank=filterbank, version=version)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # 2. Initialize Model
            model = PitchSNN().to(DEVICE)
            
            # 3. Train
            train_network(model, train_loader)
            
            # 4. Evaluate and Plot
            evaluate_noise_robustness(model, filterbank=filterbank, version=version)
        case "full_piano":
            train_on_piano_dataset(
                dataset_path='data/piano_dataset',
                version=version,
                batch_size=64,
                epochs=20,
                learning_rate=2e-4,
                device=DEVICE
            )
        case _ :
            raise NotImplementedError(f"Set proper {train_regime}")
    
if __name__ == "__main__":
    main("full_piano", "handy")
import torch
import torch.nn as nn
import snntorch as snn
# from snntorch import spikegen
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # optional – makes heatmaps prettier
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import src.generate_piano_dataset as gpd

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

class STDPWTA(nn.Module):
    """
    STDP layer with winner‑take‑all competition.
    - Input: (time, batch, input_channels) spike trains.
    - Output: (time, batch, num_output) spike trains from LIF neurons.
    - Weight updates: classical STDP (all‑to‑all) applied only to the winner neuron.
    """
    def __init__(self, num_inputs, num_outputs, 
                 tau_pre=20.0, tau_post=20.0, lr=0.002,
                 beta=0.95, threshold=1.0):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.lr = lr

        # Plastic weights (output x input)
        self.w = nn.Parameter(torch.rand(num_outputs, num_inputs) * 0.1, requires_grad=False)

        # LIF output neurons
        self.lif = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")

    def forward(self, input_spikes, win_idx=None):
        """
        input_spikes: (T, B, N_in)
        win_idx: (B,) indices of winners for this batch (used for weight update).
                 If None, no weight update is performed (evaluation mode).
        Returns output spikes (T, B, N_out)
        """
        T, B, _ = input_spikes.shape
        device = input_spikes.device

        # Synaptic currents (linear combination)
        cur = input_spikes @ self.w.T  # (T, B, N_out)

        # LIF dynamics
        mem = self.lif.init_leaky().to(device)
        spk_rec = []
        for t in range(T):
            if t > 0:
                # inhibition strength: 0.5, can be tuned
                inhibition = spk_rec[-1].sum(dim=1, keepdim=True) * 0.5   # (B, 1)
                cur[t] = cur[t] - inhibition.expand(-1, self.num_outputs)
            spk, mem = self.lif(cur[t], mem)
            spk_rec.append(spk)

        spk_rec = torch.stack(spk_rec)  # (T, B, N_out)

        # STDP weight update (only if win_idx provided)
        if win_idx is not None:
            self._stdp_update(input_spikes, spk_rec, win_idx)

        return spk_rec

    def _stdp_update(self, pre_spikes, post_spikes, win_idx):
        """
        Manual STDP update for the winner neurons only.
        pre_spikes: (T, B, N_in)
        post_spikes: (T, B, N_out)
        win_idx: (B,) long tensor
        """
        T, B, N_in = pre_spikes.shape
        N_out = post_spikes.shape[2]
        device = pre_spikes.device

        # Accumulate weight changes over time
        dw = torch.zeros_like(self.w)  # (N_out, N_in)

        for b in range(B):
            winner = win_idx[b].item()
            pre_trace = torch.zeros(N_in, device=device)
            post_trace = torch.zeros(N_out, device=device)

            for t in range(T):
                pre_spk = pre_spikes[t, b]       # (N_in)
                post_spk = post_spikes[t, b]     # (N_out)

                # Update traces (exponential decay)
                pre_trace = pre_trace * np.exp(-1.0/self.tau_pre) + pre_spk.float()
                post_trace = post_trace * np.exp(-1.0/self.tau_post) + post_spk.float()

                # STDP: dw = η (post_spike * pre_trace - pre_spike * post_trace)
                # Only for the winner's synapses
                dw[winner] += self.lr * (
                    post_spk[winner] * pre_trace - pre_spk * post_trace[winner]
                )

        # Apply the accumulated weight changes
        self.w.data += dw
        # Optional: keep weights non‑negative
        self.w.data.clamp_(min=0.0)
        # Homeostatic normalisation: each neuron's weight vector has unit L2 norm
        norms = self.w.data.norm(p=2, dim=1, keepdim=True) + 1e-8
        self.w.data = self.w.data / norms

    # def forward(self, input_spikes, win_idx=None):
    #     """
    #     input_spikes: (time, batch, num_inputs) binary spike tensor.
    #     win_idx: (batch,) tensor of winner neuron indices. If None, no STDP update.
    #     """
    #     # Synaptic currents: (time, batch, num_outputs)
    #     cur = self.fc(input_spikes)  # linear combination of inputs

    #     # Initialize membrane potential
    #     mem = self.lif.init_leaky()
    #     spk_rec = []

    #     # Simulate over time
    #     for t in range(input_spikes.size(0)):
    #         spk, mem = self.lif(cur[t], mem)
    #         spk_rec.append(spk)

    #     spk_rec = torch.stack(spk_rec)  # (time, batch, num_outputs)

    #     # Apply STDP update only for the winning neurons
    #     if win_idx is not None:
    #         # Prepare batch indices for gathering pre/post spikes
    #         batch_size = input_spikes.size(1)
    #         # For each batch, we only update the winner's synapses.
    #         # snn.STDP expects pre and post spike tensors. We can manually apply updates.
    #         # Alternatively, we can zero out the gradients of non-winners. 
    #         # Simpler: use the update method provided by the STDP object after simulation.
    #         # However, snn.STDP.stdp_update() works on the whole layer; we can mask the learning rate.
    #         # We'll set a per-neuron learning rate mask.
    #         lr_mask = torch.zeros(batch_size, self.num_outputs, device=input_spikes.device)
    #         lr_mask[torch.arange(batch_size), win_idx] = 1.0

    #         # Temporarily override learning rate to only update winners
    #         orig_lr = self.stdp.learning_rate
    #         # The STDP object stores learning_rate; we can't easily apply a mask without modification.
    #         # A clean way: call stdp.stdp_update manually with the spike records and a mask.
    #         # We'll implement a custom update loop using stdp's pre/post trace tensors.
    #         self._update_stdp_wta(input_spikes, spk_rec, win_idx)

    #     return spk_rec

    # def _update_stdp_wta(self, pre_spikes, post_spikes, win_idx):
    #     """Manually compute STDP weight updates only for winning neurons."""
    #     # pre_spikes: (T, B, in), post_spikes: (T, B, out)
    #     T, B, N_in = pre_spikes.shape
    #     _, _, N_out = post_spikes.shape

    #     # Compute pre and post traces (exponential decay)
    #     tau_pre = self.stdp.tau_pre
    #     tau_post = self.stdp.tau_post
    #     device = pre_spikes.device

    #     # Eligibility traces (B, N_in, N_out) - accumulated over time
    #     # For simplicity we use a loop over time, though it's slow for long sequences.
    #     # For our short sequences (100 ms) it's fine.
    #     dw = torch.zeros_like(self.fc.weight)  # (N_out, N_in)

    #     # For each batch separately
    #     for b in range(B):
    #         pre_trace = torch.zeros(N_in, device=device)
    #         post_trace = torch.zeros(N_out, device=device)
    #         w_idx = win_idx[b]  # winner neuron index

    #         for t in range(T):
    #             pre_spk = pre_spikes[t, b]      # (N_in)
    #             post_spk = post_spikes[t, b]    # (N_out)

    #             # Update traces
    #             pre_trace = pre_trace * np.exp(-1.0 / tau_pre) + pre_spk.float()
    #             post_trace = post_trace * np.exp(-1.0 / tau_post) + post_spk.float()

    #             # STDP: weight change = (post_spike * pre_trace) - (pre_spike * post_trace)
    #             # Only for the winner neuron
    #             dw_winner = self.lr * (
    #                 post_spk[w_idx] * pre_trace - 
    #                 pre_spk * post_trace[w_idx]
    #             )
    #             dw[w_idx] += dw_winner

    #     # Apply weight change
    #     self.fc.weight.data += dw

    # def get_weights(self):
    #     return self.fc.weight.data.clone()

# ---------- Training Function ----------
def train_stdp_wta(model, dataloader, version="spikify", epochs=10, device='cpu'):
    """
    Train the STDP-WTA network.
    dataloader yields (spike_batch, labels) where spike_batch is (batch, time, channels).
    Labels are used only for evaluation, not for training (unsupervised).
    """
    model.train()
    spike_counts = []

    for epoch in range(epochs):
        epoch_spikes = 0
        epoch_samples = 0
        for data, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # Convert to (time, batch, channels)
            data = data.transpose(0, 1).to(device)  # (T, B, C)
            B = data.size(1)

            # First forward pass (no weight update) to determine winners
            with torch.no_grad():
                spk_rec = model(data, win_idx=None)

            # # Determine winner: neuron with the earliest spike time
            # first_spk_time = torch.full((B, model.num_outputs), float('inf'), device=device)
            # for n in range(model.num_outputs):
            #     # time of first spike for neuron n in each sample
            #     spk_times = (spk_rec[:, :, n] > 0).float().argmax(dim=0)  # (B)
            #     has_spike = spk_rec[:, :, n].sum(0) > 0
            #     first_spk_time[has_spike, n] = spk_times[has_spike].float()

            # win_idx = first_spk_time.argmin(dim=1)  # (B)
            # # For samples with no spikes at all, fall back to neuron 0 (weights will not update anyway)
            # no_spike_mask = (first_spk_time == float('inf')).all(dim=1)
            # win_idx[no_spike_mask] = 0

            # Winner: neuron with the highest total spike count over the whole sample
            total_spikes = spk_rec.sum(dim=0)          # (B, N_out)
            win_idx = total_spikes.argmax(dim=1)       # (B)

            # Second forward pass WITH STDP weight update for winners
            # (Inside model.forward, win_idx triggers _stdp_update)
            _ = model(data, win_idx=win_idx)

            # Log spike counts
            epoch_spikes += spk_rec.sum().item()
            epoch_samples += B

        avg_spikes = epoch_spikes / epoch_samples
        spike_counts.append(avg_spikes)
        print(f"Epoch {epoch+1}: avg spikes/sample = {avg_spikes:.2f}")

    # ----- SAVE MODEL & METRICS -----
    save_dir = ROOT / "data" / "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f'model_stdp_wta_{version}.pth')
    print(f"Model saved to {save_dir / f'model_stdp_wta_{version}.pth'}")

    # Plot spike count evolution
    plt.figure()
    plt.plot(range(1, epochs+1), spike_counts, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Avg spikes per sample')
    plt.title('STDP Training Progress')
    plt.grid(True)
    plot_dir = ROOT / "data" / "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir / f'stdp_spike_count_{version}.pdf')
    plt.show()

    return spike_counts   # optionally return for later analysis
    

# ---------- Evaluation ----------
def evaluate_assignment(model, dataloader, version="spikify", device='cpu'):
    """
    After training, assign each output neuron to the class it fires most for.
    Then compute classification accuracy.
    """
    model.eval()
    # neuron_spikes_per_label on CPU is fine – we index with .item() later
    neuron_spikes_per_label = torch.zeros(model.num_outputs, model.num_outputs)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.transpose(0, 1).to(device)
            spk_rec = model(data, win_idx=None)
            total_spikes = spk_rec.sum(dim=0)          # (batch, out) on device
            _, winners = total_spikes.max(dim=1)       # (batch,) on device
            for b in range(len(labels)):
                # .item() moves the scalar to CPU – safe
                neuron_spikes_per_label[winners[b].item(), labels[b].item()] += 1

    # Assign each neuron to the label it fires most for (majority vote)
    neuron_assignments = neuron_spikes_per_label.argmax(dim=1)   # still CPU

    torch.save({'neuron_assignments': neuron_assignments,
                'neuron_spikes_per_label': neuron_spikes_per_label},
               str(ROOT / "data" / "trained_models" / f'stdp_neuron_assignments_{version}.pt'))
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.transpose(0, 1).to(device)
            spk_rec = model(data)
            total_spikes = spk_rec.sum(dim=0)
            _, winners = total_spikes.max(dim=1)        # on device

            # Move winners to CPU before indexing CPU tensor
            winners_cpu = winners.cpu()
            predicted_labels = neuron_assignments[winners_cpu]   # now safe

            all_preds.append(predicted_labels)
            all_labels.append(labels)                   # labels already CPU
            correct += (predicted_labels == labels).sum().item()
            total += len(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = 100 * correct / total
    print(f"Test accuracy (after neuron assignment): {acc:.2f}%")

    key_names = gpd.piano_key_labels()
    plot_confusion(all_labels, all_preds, classes=model.num_outputs,
                   title='STDP-WTA Confusion Matrix (After Assignment)',
                   save_path=str(ROOT / "data" / "plots" / f'confusion_stdp_{version}.pdf'),
                   label_names=key_names)
    return neuron_assignments

def load_piano_dataset(dataset_path, batch_size=64):
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
    return train_loader, test_loader


# ---------- Usage Example ----------
if __name__ == "__main__":
    version = "handy"
    # Assume you already have a DataLoader with piano spike data
    train_loader, test_loader = load_piano_dataset(
        dataset_path=str(ROOT / 'data' / f'piano_dataset_{version}.pt'),
        batch_size=64
        )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trained with {DEVICE}:")
    # Model: input = cochlear channels (e.g., 50), output = number of pitches (88)
    model = STDPWTA(num_inputs=50, num_outputs=88, lr=0.002).to(DEVICE)
    train_stdp_wta(model, train_loader, epochs=15, version=version, device=DEVICE)
    neuron_labels = evaluate_assignment(model, test_loader, version=version, device=DEVICE)
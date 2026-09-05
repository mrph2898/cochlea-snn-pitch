"""
SNN Architecture for Pitch Classification using snnTorch.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class PitchSNN(nn.Module):
    def __init__(self, num_inputs=50, num_hidden=100, num_outputs=3, beta=0.9):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Layer 1: Hidden layer (Spiking)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 2: Output layer (Non-spiking, pure leaky integrator for logits)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        # Setting threshold very high so it never fires a spike or resets its voltage
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1e9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        mem2_rec = []  # We only need to record membrane potentials for the loss
        
        time_steps = x.size(0)
        
        for step in range(time_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2) # We discard output spikes
            
            mem2_rec.append(mem2)
            
        return torch.stack(mem2_rec) # Shape: (Time, Batch, Classes)
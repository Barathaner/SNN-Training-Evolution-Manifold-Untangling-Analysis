import snntorch as snn
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs, num_steps, beta):
        super().__init__()

        # Layer 0: Input -> Hidden0 (gleiche Größe wie Input)
        self.fc0 = nn.Linear(num_inputs, num_inputs)
        self.lif0 = snn.Leaky(beta=beta)
        
        # Erste Hidden Layer: Hidden0 -> Hidden1
        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        
        # Zweite Hidden Layer: Hidden1 -> Hidden2 (hierarchisch)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        
        # Output Layer: Hidden2 -> Output
        self.fc3 = nn.Linear(num_hidden2, num_outputs)
        self.lif3 = snn.Leaky(beta=beta)
        
        self.num_steps = num_steps

    def forward(self, x):
        # x shape: [B, T, num_inputs]
        B, T, _ = x.shape
        assert T == self.num_steps, f"Eingabezeitdimension ({T}) stimmt nicht mit num_steps ({self.num_steps}) überein."

        # Initialisiere hidden states für alle Layer
        mem0 = self.lif0.reset_mem()
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()

        # Recording-Listen nur für Output-Layer (für Return-Wert)
        spk3_rec = []
        mem3_rec = []

        for step in range(T):
            x_t = x[:, step, :]              # Shape: [B, num_inputs]
            
            # Layer 0: Input -> Hidden0 (gleiche Größe)
            cur0 = self.fc0(x_t)
            spk0, mem0 = self.lif0(cur0, mem0)
            
            # Layer 1: Hidden0 -> Hidden1
            cur1 = self.fc1(spk0)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2: Hidden1 -> Hidden2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3: Hidden2 -> Output
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Speichere nur Output-Layer Aktivitäten (für Return-Wert)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        # [T, B, features] → [B, T, features]
        spk3_bt = torch.stack(spk3_rec, dim=0).permute(1, 0, 2)
        mem3_bt = torch.stack(mem3_rec, dim=0).permute(1, 0, 2)

        return spk3_bt, mem3_bt

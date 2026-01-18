import snntorch as snn
import torch.nn as nn
import torch

class RSNN(nn.Module):
    """
    Rekurrentes Spiking Neural Network (RSNN) mit RLeaky Neuronen.
    
    Architektur basierend auf Cramer et al. (2020) und ähnlichen Arbeiten:
    Input -> Recurrent Hidden Layer -> Output
    
    Das rekurrente Hidden Layer verwendet RLeaky Neuronen mit all-to-all Verbindungen,
    was dem Modell ermöglicht, zeitliche Abhängigkeiten zu lernen.
    
    Args:
        num_inputs: Anzahl der Input-Features
        num_hidden: Anzahl der Neuronen im rekurrenten Hidden Layer
        num_outputs: Anzahl der Output-Klassen
        num_steps: Anzahl der Zeitschritte
        beta: Leaky-Integrate-and-Fire Decay-Faktor (typischerweise 0.9-0.99)
    """
    
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta=0.9):
        super().__init__()
        
        # Input -> Recurrent Hidden Layer
        # Linear Layer transformiert Input zu Hidden-Dimension
        self.fc_input = nn.Linear(num_inputs, num_hidden)
        
        # Rekurrentes Hidden Layer mit RLeaky Neuronen
        # all_to_all=True bedeutet: Jedes Neuron ist mit jedem anderen im Hidden Layer verbunden
        # linear_features=num_hidden definiert die Größe der rekurrenten Gewichte
        self.rlif = snn.RLeaky(
            beta=beta, 
            all_to_all=True, 
            linear_features=num_hidden,
            reset_mechanism="zero"
        )
        
        # Recurrent Hidden -> Output Layer
        self.fc_output = nn.Linear(num_hidden, num_outputs)
        
        # Output Layer mit einfachem Leaky (keine Rekurrenz)
        self.lif_output = snn.RLeaky(beta=beta, output=True,reset_mechanism="zero",V=0.5,all_to_all=False)
        
        self.num_steps = num_steps
        self.num_hidden = num_hidden
        

    def forward(self, x):
        """
        Forward-Pass durch das rekurrente SNN.
        
        Args:
            x: Input Tensor mit Shape [B, T, num_inputs]
               B = Batch-Größe
               T = Anzahl Zeitschritte
               num_inputs = Anzahl Input-Features
        
        Returns:
            spk_output: Spike-Ausgaben des Output-Layers [B, T, num_outputs]
            mem_output: Membranpotenzial des Output-Layers [B, T, num_outputs]
        """
        # x shape: [B, T, num_inputs]
        B, T, _ = x.shape
        
        # Automatische Anpassung der Zeitdimension falls nötig
        if T != self.num_steps:
            if T > self.num_steps:
                # Schneide ab, wenn zu viele Zeitschritte
                x = x[:, :self.num_steps, :]
                T = self.num_steps
            else:
                # Padde mit Nullen, wenn zu wenige Zeitschritte
                padding = torch.zeros(B, self.num_steps - T, x.shape[2], 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
                T = self.num_steps
        
        # Initialisiere hidden states für rekurrentes Layer
        # RLeaky benötigt init_rleaky() für Spikes und Membranpotenzial
        spk_hidden, mem_hidden = self.rlif.init_rleaky()
        spk_output, mem_output = self.lif_output.init_rleaky()
        
        
        # Recording-Listen für Output-Layer
        spk_output_rec = []
        mem_output_rec = []
        
        # Zeitlicher Forward-Pass
        for step in range(T):
            x_t = x[:, step, :]  # Shape: [B, num_inputs]
            
            # Input -> Recurrent Hidden Layer
            # Transformiere Input zu Hidden-Dimension
            cur_hidden = self.fc_input(x_t)  # Shape: [B, num_hidden]
            
            # Rekurrentes Hidden Layer
            # RLeaky benötigt: (current_input, previous_spikes, previous_membrane)
            # Die Rekurrenz wird intern durch RLeaky gehandhabt
            spk_hidden, mem_hidden = self.rlif(cur_hidden, spk_hidden, mem_hidden)
            
            # Recurrent Hidden -> Output Layer
            cur_output = self.fc_output(spk_hidden)  # Shape: [B, num_outputs]
            
            # Output Layer (keine Rekurrenz)
            spk_output, mem_output = self.lif_output(cur_output, mem_output)
            
            # Speichere Output-Layer Aktivitäten
            spk_output_rec.append(spk_output)
            mem_output_rec.append(mem_output)
        
        # Staple alle Zeitschritte: [T, B, features] → [B, T, features]
        spk_output_bt = torch.stack(spk_output_rec, dim=0).permute(1, 0, 2)
        mem_output_bt = torch.stack(mem_output_rec, dim=0).permute(1, 0, 2)
        
        return spk_output_bt, mem_output_bt


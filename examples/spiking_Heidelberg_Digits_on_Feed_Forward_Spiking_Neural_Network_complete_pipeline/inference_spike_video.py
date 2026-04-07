"""
Skript zur Inferenz mit einem trainierten SNN-Modell und Erstellung eines Spike-Videos.
Lädt Modellgewichte, führt Inferenz für ein Sample durch und visualisiert die Spikes
aus der Readout-Layer als animiertes Video.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

# Projekt-Imports
import models.sffnn_batched as sffnn_batched
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader

# Device-Konfiguration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Device: {device}")

# Projekt-Root bestimmen
project_root = os.path.abspath(os.path.dirname(__file__))
model_weights_path = os.path.join(project_root, "models", "model_export", "model_weights.pth")
data_path = os.path.join(project_root, "data", "input")

# Modell-Parameter (aus Komplette_Konfiguration.md)
num_inputs = 350
num_hidden1 = 128
num_hidden2 = 64
num_outputs = 10
num_steps = 80
beta = 0.9

# Preprocessing-Parameter (aus Komplette_Konfiguration.md)
n_time_bins = 80
target_neurons = 350
original_neurons = 700
fixed_duration = 958007.0
gaussian_sigma = 1.0

def main():
    # 1. Modell erstellen und Gewichte laden
    print("📦 Lade Modell...")
    net = sffnn_batched.Net(
        num_inputs=num_inputs,
        num_hidden1=num_hidden1,
        num_hidden2=num_hidden2,
        num_outputs=num_outputs,
        num_steps=num_steps,
        beta=beta
    ).to(device)
    
    # Gewichte laden
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Modellgewichte nicht gefunden: {model_weights_path}")
    
    net.load_state_dict(torch.load(model_weights_path, map_location=device))
    net.eval()
    print(f"✅ Modell geladen von: {model_weights_path}")
    
    # 2. Preprocessing-Pipeline erstellen
    print("🔧 Erstelle Preprocessing-Pipeline...")
    transform = datatransforms.get_preprocessing(
        n_time_bins=n_time_bins,
        target_neurons=target_neurons,
        original_neurons=original_neurons,
        fixed_duration=fixed_duration,
        gaussian_sigma=gaussian_sigma,
        include_trim_silence=False
    )
    
    # 3. Datensatz laden (nur ein Sample)
    print("📊 Lade Datensatz...")
    dataloader_obj = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10),
        data_path=data_path,
        transform=transform,
        train=False,  # Test-Set verwenden
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    # Ein Sample aus dem DataLoader holen
    sample, label = next(iter(dataloader_obj))
    sample = sample.to(device)
    label = label.item()
    
    print(f"✅ Sample geladen - Label: {label}")
    print(f"   Sample Shape: {sample.shape}")  # Sollte [1, 80, 350] sein
    
    # 4. Inferenz durchführen und Spikes aus Readout-Layer aufzeichnen
    print("🧠 Führe Inferenz durch...")
    
    # Initialisiere Memory-States
    mem0 = net.lif0.reset_mem()
    mem1 = net.lif1.reset_mem()
    mem2 = net.lif2.reset_mem()
    mem3 = net.lif3.reset_mem()
    
    # Recording-Listen für Readout-Layer (lif3)
    spk3_rec = []
    mem3_rec = []
    
    # Manuelle Simulation (ähnlich wie im Beispiel)
    with torch.no_grad():
        for step in range(num_steps):
            x_t = sample[:, step, :]  # Shape: [1, 350]
            
            # Forward-Pass durch alle Layer
            cur0 = net.fc0(x_t)
            spk0, mem0 = net.lif0(cur0, mem0)
            
            cur1 = net.fc1(spk0)
            spk1, mem1 = net.lif1(cur1, mem1)
            
            cur2 = net.fc2(spk1)
            spk2, mem2 = net.lif2(cur2, mem2)
            
            cur3 = net.fc3(spk2)
            spk3, mem3 = net.lif3(cur3, mem3)
            
            # Speichere Spikes aus Readout-Layer
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
    
    # Konvertiere Listen zu Tensoren
    # spk3_rec: Liste von [1, 10] Tensoren -> [80, 1, 10]
    spk3_rec = torch.stack(spk3_rec)  # Shape: [80, 1, 10]
    
    print(f"✅ Inferenz abgeschlossen")
    print(f"   Spike-Recording Shape: {spk3_rec.shape}")
    
    # 5. Video erstellen
    print("🎬 Erstelle Spike-Video...")
    
    # Vorbereitung für Visualisierung
    # spk3_rec: [80, 1, 10] -> [80, 10] (Batch-Dimension entfernen)
    spk3_rec = spk3_rec.squeeze(1).detach().cpu()  # Shape: [80, 10]
    
    # Labels für die 10 Klassen
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Erstelle Figure und Axes
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    
    # Erstelle Animation mit snntorch.spikeplot
    anim = splt.spike_count(spk3_rec, fig, ax, labels=labels, animate=True)
    
    # Speichere Video
    output_path = os.path.join(project_root, "spike_bar.mp4")
    print(f"💾 Speichere Video nach: {output_path}")
    anim.save(output_path, writer='ffmpeg', fps=10)
    
    print(f"✅ Video erfolgreich erstellt: {output_path}")
    print(f"   Vorhergesagtes Label: {torch.sum(spk3_rec, dim=0).argmax().item()}")
    print(f"   Tatsächliches Label: {label}")
    
    # Optional: HTML-Video für Jupyter Notebook
    # HTML(anim.to_html5_video())

if __name__ == "__main__":
    main()

"""
Wrapper für Mean-Field-Theoretic Manifold Analysis.

Einfache Funktion zur Analyse von Manifold-Metriken aus einem DataLoader.
"""

import numpy as np
from typing import List, Dict, Union, Optional
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

from manifolduntanglinganalysis.mftma.manifold_analysis_correlation import manifold_analysis_corr


def analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
    dataloader,
    labels: List[int],
    max_samples_per_class: int = 100,
    kappa: float = 0.0,
    n_t: int = 200,
    n_reps: int = 1,
    verbose: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Führt Mean-Field-Theoretic Manifold Analysis auf einem DataLoader durch.
    
    Args:
        dataloader: PyTorch DataLoader mit preprocesseten Daten
                   Erwartet: (frames, label) oder (data, label) Tupel
                   Frames sollten Shape (n_time_bins, 1, n_neurons) oder (n_time_bins, n_neurons) haben
        labels: Liste von Labels, die analysiert werden sollen (z.B. [0, 1, 2, ..., 9])
        max_samples_per_class: Maximale Anzahl Samples pro Klasse (default: 100)
        kappa: Margin size für die Analyse (default: 0.0)
        n_t: Anzahl der Gaussian-Vektoren pro Manifold (default: 200)
        n_reps: Anzahl der Wiederholungen für Korrelationsanalyse (default: 1)
        verbose: Wenn True, werden Fortschrittsinformationen ausgegeben (default: True)
    
    Returns:
        Dictionary mit folgenden Metriken:
            - 'capacity': Durchschnittliche Capacity (α_M)
            - 'radius': Durchschnittlicher Radius (R_M)
            - 'dimension': Durchschnittliche Dimension (D_M)
            - 'correlation': Center Correlation
            - 'K': Optimal K
            - 'capacity_per_class': Capacity pro Klasse (Array)
            - 'radius_per_class': Radius pro Klasse (Array)
            - 'dimension_per_class': Dimension pro Klasse (Array)
    """
    if len(labels) < 2:
        raise ValueError(f"Mindestens 2 Labels benötigt, aber nur {len(labels)} gegeben")
    
    if verbose:
        print(f"Analysiere {len(labels)} Klassen mit max. {max_samples_per_class} Samples pro Klasse")
    
    X_by_class = []
    samples_per_class = {}
    n_neurons = None
    
    for label in labels:
        all_time_bins = []
        samples_found = 0
        
        if verbose:
            print(f"  Sammle Daten für Label {label}...", end=" ", flush=True)
        
        for batch_data, batch_labels in dataloader:
            if samples_found >= max_samples_per_class:
                break
            
            batch_labels_np = batch_labels.numpy() if hasattr(batch_labels, 'numpy') else np.array(batch_labels)
            batch_data_np = batch_data.numpy() if hasattr(batch_data, 'numpy') else np.array(batch_data)
            
            for i, lbl in enumerate(batch_labels_np):
                if samples_found >= max_samples_per_class:
                    break
                
                if lbl == label:
                    frames = batch_data_np[i]
                    
                    if frames.ndim == 3:
                        vec = frames[:, 0, :]
                    elif frames.ndim == 2:
                        vec = frames
                    else:
                        raise ValueError(f"Unerwartete Frame-Shape: {frames.shape}")
                    
                    if n_neurons is None:
                        n_neurons = vec.shape[1]
                    
                    for t in range(vec.shape[0]):
                        all_time_bins.append(vec[t, :])
                    
                    samples_found += 1
        
        if len(all_time_bins) > 0:
            trajectory_data = np.array(all_time_bins).T
            X_by_class.append(trajectory_data)
            samples_per_class[label] = samples_found
            if verbose:
                print(f"✓ {samples_found} Samples, {trajectory_data.shape[1]} Time-Bins")
        else:
            if verbose:
                print(f"✗ Keine Daten gefunden")
            n_neurons = n_neurons if n_neurons is not None else 350
            X_by_class.append(np.zeros((n_neurons, 0)))
    
    X = [x for x in X_by_class if x.shape[1] > 0]
    
    if len(X) < 2:
        raise ValueError(f"Nicht genug Daten: Nur {len(X)} Klassen mit Daten gefunden")
    
    if verbose:
        print(f"\nFühre Manifold-Analyse durch...")
        print(f"  {len(X)} Klassen mit Daten")
        for i, x in enumerate(X):
            print(f"    Klasse {labels[i]}: {x.shape[1]} Time-Bins, {x.shape[0]} Neuronen")
    
    capacity_all, radius_all, dimension_all, correlation_all, K_all = manifold_analysis_corr(
        X, kappa, n_t, n_reps=n_reps
    )
    
    avg_capacity = 1 / np.mean(1 / capacity_all)
    avg_radius = np.mean(radius_all)
    avg_dimension = np.mean(dimension_all)
    
    results = {
        'capacity': avg_capacity,
        'radius': avg_radius,
        'dimension': avg_dimension,
        'correlation': correlation_all,
        'K': K_all,
        'capacity_per_class': capacity_all,
        'radius_per_class': radius_all,
        'dimension_per_class': dimension_all,
        'samples_per_class': samples_per_class
    }
    
    if verbose:
        print(f"\nErgebnisse:")
        print(f"  Capacity (α_M):  {avg_capacity:.4f}")
        print(f"  Radius (R_M):    {avg_radius:.4f}")
        print(f"  Dimension (D_M): {avg_dimension:.4f}")
        print(f"  Correlation:     {correlation_all:.4f}")
        print(f"  Optimal K:       {K_all}")
    
    return results


def plot_manifold_metrics_over_epochs(
    results: List[Dict[str, Union[float, np.ndarray]]],
    activity_logs_path: str,
    input_data_metrics: Optional[Dict[str, float]] = None,
    save_dir: Optional[str] = None,
    figsize_per_subplot: tuple = (5, 4)
) -> plt.Figure:
    """
    Erstellt einen Plot für Manifold-Metriken (Capacity, Radius, Dimension) über Epochen.
    
    Der Plot zeigt Layer in Zeilen und Metriken in Spalten. Jeder Subplot zeigt die
    Entwicklung einer Metrik für einen Layer über die Epochen. Optional können Baseline-Werte
    aus Input-Daten als gestrichelte Linien angezeigt werden.
    
    Args:
        results: Liste von Ergebnis-Dictionaries (jeweils mit 'capacity', 'radius', 'dimension')
        activity_logs_path: Pfad zum Verzeichnis mit Activity-Log-Dateien
        input_data_metrics: Optional, Dictionary mit Baseline-Metriken für Input-Daten.
                          Format: {'capacity': float, 'radius': float, 'dimension': float}
                          Wird als gestrichelte rote Linie in jedem Plot angezeigt.
        save_dir: Optional, Verzeichnis zum Speichern des Plots. Wenn None, wird Plot nicht gespeichert.
        figsize_per_subplot: Größe pro Subplot (default: (5, 4))
    
    Returns:
        matplotlib.pyplot.Figure: Die erstellte Figure
    
    Die Funktion erwartet, dass die Dateinamen im Format sind:
        epoch_{epoch:03d}_{layer_name}_spk_events.h5
    """
    activity_logs_path = Path(activity_logs_path)
    
    if not activity_logs_path.exists():
        raise ValueError(f"Activity logs Pfad existiert nicht: {activity_logs_path}")
    
    # Lade alle H5-Dateien
    h5_files = sorted(activity_logs_path.glob("*.h5"))
    
    if len(h5_files) != len(results):
        raise ValueError(
            f"Anzahl der H5-Dateien ({len(h5_files)}) stimmt nicht mit Anzahl der Ergebnisse ({len(results)}) überein"
        )
    
    # Parse Dateinamen und gruppiere nach Layer
    layer_data = {}  # {layer_name: {epoch: {metric: value}}}
    
    # Regex-Pattern für Dateinamen: epoch_XXX_layername_spk_events.h5
    pattern = re.compile(r'epoch_(\d+)_(\w+)_spk_events\.h5')
    
    for h5_file, result in zip(h5_files, results):
        match = pattern.match(h5_file.name)
        if not match:
            print(f"⚠️ Warnung: Dateiname passt nicht zum erwarteten Format: {h5_file.name}")
            continue
        
        epoch = int(match.group(1))
        layer_name = match.group(2)
        
        if layer_name not in layer_data:
            layer_data[layer_name] = {}
        
        layer_data[layer_name][epoch] = {
            'capacity': result['capacity'],
            'radius': result['radius'],
            'dimension': result['dimension']
        }
    
    if not layer_data:
        raise ValueError("Keine gültigen Layer-Daten gefunden. Überprüfe die Dateinamen.")
    
    # Sortiere Layer für konsistente Reihenfolge
    layer_names = sorted(layer_data.keys())
    n_layers = len(layer_names)
    
    # Erstelle ein großes Subplot-Grid: n_layers Zeilen × 3 Spalten
    fig, axes = plt.subplots(
        n_layers, 3, 
        figsize=(figsize_per_subplot[0] * 3, figsize_per_subplot[1] * n_layers)
    )
    
    # Falls nur ein Layer, mache axes zu 2D-Array
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Manifold Metriken über Epochen', fontsize=16, fontweight='bold', y=1.0)
    
    # Metriken-Namen und Farben
    metric_names = ['Capacity (α_M)', 'Radius (R_M)', 'Dimension (D_M)']
    metric_keys = ['capacity', 'radius', 'dimension']
    colors = ['blue', 'orange', 'green']
    markers = ['o', 's', '^']
    
    # Plotte für jeden Layer
    for row_idx, layer_name in enumerate(layer_names):
        epochs_data = layer_data[layer_name]
        epochs = sorted(epochs_data.keys())
        
        # Extrahiere Metriken für diesen Layer
        metric_values = {
            'capacity': [epochs_data[ep]['capacity'] for ep in epochs],
            'radius': [epochs_data[ep]['radius'] for ep in epochs],
            'dimension': [epochs_data[ep]['dimension'] for ep in epochs]
        }
        
        # Plotte jede Metrik in einer Spalte
        for col_idx, (metric_key, metric_name, color, marker) in enumerate(
            zip(metric_keys, metric_names, colors, markers)
        ):
            ax = axes[row_idx, col_idx]
            
            # Plot der Layer-Daten
            ax.plot(
                epochs, metric_values[metric_key],
                marker=marker, linestyle='-', linewidth=2,
                markersize=6, color=color, label=layer_name
            )
            
            # Baseline-Linie (gestrichelt) falls vorhanden
            if input_data_metrics is not None and metric_key in input_data_metrics:
                baseline_value = input_data_metrics[metric_key]
                ax.axhline(
                    y=baseline_value,
                    color='red', linestyle='--', linewidth=2,
                    alpha=0.7, label='Input Data (Baseline)'
                )
            
            # Labels und Titel
            if row_idx == 0:  # Nur in der ersten Zeile Titel setzen
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
            
            if row_idx == n_layers - 1:  # Nur in der letzten Zeile X-Label
                ax.set_xlabel('Epoche', fontsize=11)
            
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_xticks(epochs)
            ax.grid(True, alpha=0.3)
            
            # Layer-Name als Y-Label auf der linken Seite
            if col_idx == 0:
                ax.text(-0.15, 0.5, layer_name, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', rotation=90,
                       ha='center', va='center')
            
            # Legende nur im ersten Subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Speichere Plot falls gewünscht
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = "manifold_metrics_all_layers.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Plot gespeichert: {filepath}")
    
    return fig


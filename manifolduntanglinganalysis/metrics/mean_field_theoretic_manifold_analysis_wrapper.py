"""
Wrapper für Mean-Field-Theoretic Manifold Analysis.

Einfache Funktion zur Analyse von Manifold-Metriken aus einem DataLoader.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple
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


def analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded(
    dataloader,
    labels: List[int],
    max_samples_per_class: int = 100,
    kappa: float = 0.0,
    n_t: int = 200,
    n_reps: int = 1,
    verbose: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Führt Mean-Field-Theoretic Manifold Analysis auf einem DataLoader durch,
    speziell für rate-coded Output-Layer.
    
    Bei rate-coded Output-Layern wird die kumulative Summe (cumsum) der Spikes verwendet,
    da das Readout-Layer so trainiert wird. Die Manifold-Analyse wird daher auf der
    kumulativen Aktivität statt auf den rohen Frames durchgeführt.
    
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
        print(f"Analysiere {len(labels)} Klassen mit max. {max_samples_per_class} Samples pro Klasse (Rate-Coded)")
    
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
                    
                    # Normalisiere Frame-Shape zu (n_time_bins, n_neurons)
                    if frames.ndim == 3:
                        vec = frames[:, 0, :]
                    elif frames.ndim == 2:
                        vec = frames
                    else:
                        raise ValueError(f"Unerwartete Frame-Shape: {frames.shape}")
                    
                    if n_neurons is None:
                        n_neurons = vec.shape[1]
                    
                    # Berechne kumulative Summe entlang der Zeitachse (axis=0)
                    # Dies entspricht der rate-coded Aktivität, wie sie im Readout-Layer verwendet wird
                    vec_cumsum = np.cumsum(vec, axis=0)
                    
                    # Füge jeden Time-Bin der kumulativen Summe hinzu
                    for t in range(vec_cumsum.shape[0]):
                        all_time_bins.append(vec_cumsum[t, :])
                    
                    samples_found += 1
        
        if len(all_time_bins) > 0:
            trajectory_data = np.array(all_time_bins).T
            X_by_class.append(trajectory_data)
            samples_per_class[label] = samples_found
            if verbose:
                print(f"✓ {samples_found} Samples, {trajectory_data.shape[1]} Time-Bins (cumsum)")
        else:
            if verbose:
                print(f"✗ Keine Daten gefunden")
            n_neurons = n_neurons if n_neurons is not None else 350
            X_by_class.append(np.zeros((n_neurons, 0)))
    
    X = [x for x in X_by_class if x.shape[1] > 0]
    
    if len(X) < 2:
        raise ValueError(f"Nicht genug Daten: Nur {len(X)} Klassen mit Daten gefunden")
    
    if verbose:
        print(f"\nFühre Manifold-Analyse durch (auf cumsum)...")
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
        print(f"\nErgebnisse (Rate-Coded, cumsum):")
        print(f"  Capacity (α_M):  {avg_capacity:.4f}")
        print(f"  Radius (R_M):    {avg_radius:.4f}")
        print(f"  Dimension (D_M): {avg_dimension:.4f}")
        print(f"  Correlation:     {correlation_all:.4f}")
        print(f"  Optimal K:       {K_all}")
    
    return results


def plot_manifold_metrics_over_epochs(
    results: Union[List[Dict[str, Union[float, np.ndarray]]], Dict[tuple, Dict[str, Union[float, np.ndarray]]]],
    activity_logs_path: Optional[Union[str, List[str]]] = None,
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
        results: Entweder:
                1. Dictionary mit (epoch, layer_name) Tupeln als Keys und Ergebnis-Dictionaries als Values
                   Format: {(epoch, layer_name): {'capacity': float, 'radius': float, 'dimension': float}, ...}
                2. Liste von Ergebnis-Dictionaries (für Rückwärtskompatibilität)
        activity_logs_path: Optional, entweder:
                            - Pfad zum Verzeichnis mit Activity-Log-Dateien (wenn results eine Liste ist)
                            - Liste von Dateinamen (wenn results eine Liste ist)
                            - Wird ignoriert, wenn results ein Dictionary ist
        input_data_metrics: Optional, Dictionary mit Baseline-Metriken für Input-Daten.
                          Format: {'capacity': float, 'radius': float, 'dimension': float}
                          Wird als gestrichelte rote Linie in jedem Plot angezeigt.
        save_dir: Optional, Verzeichnis zum Speichern des Plots. Wenn None, wird Plot nicht gespeichert.
        figsize_per_subplot: Größe pro Subplot (default: (5, 4))
    
    Returns:
        matplotlib.pyplot.Figure: Die erstellte Figure
    """
    layer_data = {}  # {layer_name: {epoch: {metric: value}}}
    
    # Prüfe, ob results ein Dictionary oder eine Liste ist
    if isinstance(results, dict):
        # Neue flexible API: Dictionary mit (epoch, layer_name) -> result
        for (epoch, layer_name), result in results.items():
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            layer_data[layer_name][epoch] = {
                'capacity': result['capacity'],
                'radius': result['radius'],
                'dimension': result['dimension']
            }
    elif isinstance(results, list):
        # Alte API: Liste von Ergebnissen + activity_logs_path
        if activity_logs_path is None:
            raise ValueError("Wenn results eine Liste ist, muss activity_logs_path angegeben werden")
        
        # Prüfe, ob activity_logs_path ein Verzeichnis oder eine Liste von Dateinamen ist
        if isinstance(activity_logs_path, list):
            # Liste von Dateinamen
            file_names = activity_logs_path
        else:
            # Pfad zum Verzeichnis
            activity_logs_path = Path(activity_logs_path)
            if not activity_logs_path.exists():
                raise ValueError(f"Activity logs Pfad existiert nicht: {activity_logs_path}")
            h5_files = sorted(activity_logs_path.glob("*.h5"))
            file_names = [f.name for f in h5_files]
        
        if len(file_names) != len(results):
            raise ValueError(
                f"Anzahl der Dateien ({len(file_names)}) stimmt nicht mit Anzahl der Ergebnisse ({len(results)}) überein"
            )
        
        # Regex-Pattern für Dateinamen: epoch_XXX_layername_spk_events.h5
        pattern = re.compile(r'epoch_(\d+)_(\w+)_spk_events\.h5')
        
        for file_name, result in zip(file_names, results):
            match = pattern.match(file_name)
            if not match:
                print(f"⚠️ Warnung: Dateiname passt nicht zum erwarteten Format: {file_name}")
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
    else:
        raise TypeError(f"results muss ein Dictionary oder eine Liste sein, nicht {type(results)}")
    
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


def plot_manifold_metrics_over_epochs_all_layer_in_one_plot(
    results: Union[List[Dict[str, Union[float, np.ndarray]]], Dict[tuple, Dict[str, Union[float, np.ndarray]]]],
    activity_logs_path: Optional[Union[str, List[str]]] = None,
    input_data_metrics: Optional[Dict[str, float]] = None,
    save_dir: Optional[str] = None,
    figsize_per_subplot: tuple = (6, 4)
) -> plt.Figure:
    """
    Erstellt einen Plot für Manifold-Metriken (Capacity, Radius, Dimension) über Epochen.
    
    Der Plot zeigt alle Layer in einem Plot pro Metrik. Jeder Subplot zeigt die
    Entwicklung aller Layer für eine Metrik über die Epochen. Optional können Baseline-Werte
    aus Input-Daten als gestrichelte Linien angezeigt werden.
    
    Args:
        results: Entweder:
                1. Dictionary mit (epoch, layer_name) Tupeln als Keys und Ergebnis-Dictionaries als Values
                   Format: {(epoch, layer_name): {'capacity': float, 'radius': float, 'dimension': float}, ...}
                2. Liste von Ergebnis-Dictionaries (für Rückwärtskompatibilität)
        activity_logs_path: Optional, entweder:
                            - Pfad zum Verzeichnis mit Activity-Log-Dateien (wenn results eine Liste ist)
                            - Liste von Dateinamen (wenn results eine Liste ist)
                            - Wird ignoriert, wenn results ein Dictionary ist
        input_data_metrics: Optional, Dictionary mit Baseline-Metriken für Input-Daten.
                          Format: {'capacity': float, 'radius': float, 'dimension': float}
                          Wird als gestrichelte rote Linie in jedem Plot angezeigt.
        save_dir: Optional, Verzeichnis zum Speichern des Plots. Wenn None, wird Plot nicht gespeichert.
        figsize_per_subplot: Größe pro Subplot (default: (6, 4))
    
    Returns:
        matplotlib.pyplot.Figure: Die erstellte Figure
    """
    layer_data = {}  # {layer_name: {epoch: {metric: value}}}
    
    # Prüfe, ob results ein Dictionary oder eine Liste ist
    if isinstance(results, dict):
        # Neue flexible API: Dictionary mit (epoch, layer_name) -> result
        for (epoch, layer_name), result in results.items():
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            layer_data[layer_name][epoch] = {
                'capacity': result['capacity'],
                'radius': result['radius'],
                'dimension': result['dimension']
            }
    elif isinstance(results, list):
        # Alte API: Liste von Ergebnissen + activity_logs_path
        if activity_logs_path is None:
            raise ValueError("Wenn results eine Liste ist, muss activity_logs_path angegeben werden")
        
        # Prüfe, ob activity_logs_path ein Verzeichnis oder eine Liste von Dateinamen ist
        if isinstance(activity_logs_path, list):
            # Liste von Dateinamen
            file_names = activity_logs_path
        else:
            # Pfad zum Verzeichnis
            activity_logs_path = Path(activity_logs_path)
            if not activity_logs_path.exists():
                raise ValueError(f"Activity logs Pfad existiert nicht: {activity_logs_path}")
            h5_files = sorted(activity_logs_path.glob("*.h5"))
            file_names = [f.name for f in h5_files]
        
        if len(file_names) != len(results):
            raise ValueError(
                f"Anzahl der Dateien ({len(file_names)}) stimmt nicht mit Anzahl der Ergebnisse ({len(results)}) überein"
            )
        
        # Regex-Pattern für Dateinamen: epoch_XXX_layername_spk_events.h5
        pattern = re.compile(r'epoch_(\d+)_(\w+)_spk_events\.h5')
        
        for file_name, result in zip(file_names, results):
            match = pattern.match(file_name)
            if not match:
                print(f"⚠️ Warnung: Dateiname passt nicht zum erwarteten Format: {file_name}")
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
    else:
        raise TypeError(f"results muss ein Dictionary oder eine Liste sein, nicht {type(results)}")
    
    if not layer_data:
        raise ValueError("Keine gültigen Layer-Daten gefunden. Überprüfe die Dateinamen.")
    
    # Sortiere Layer für konsistente Reihenfolge
    layer_names = sorted(layer_data.keys())
    n_layers = len(layer_names)
    
    # Erstelle ein Subplot-Grid: 3 Zeilen × 1 Spalte (eine Metrik pro Zeile)
    fig, axes = plt.subplots(
        3, 1, 
        figsize=(figsize_per_subplot[0], figsize_per_subplot[1] * 3)
    )
    
    # Falls nur eine Metrik, mache axes zu Array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    fig.suptitle('Manifold Metriken über Epochen (Alle Layer)', fontsize=16, fontweight='bold', y=0.995)
    
    # Metriken-Namen und Farben für Baseline
    metric_names = ['Capacity (α_M)', 'Radius (R_M)', 'Dimension (D_M)']
    metric_keys = ['capacity', 'radius', 'dimension']
    baseline_color = 'red'
    
    # Farben und Marker für verschiedene Layer
    layer_colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    layer_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Sammle alle Epochen für alle Layer
    all_epochs = set()
    for layer_name in layer_names:
        all_epochs.update(layer_data[layer_name].keys())
    all_epochs = sorted(all_epochs)
    
    # Plotte jede Metrik in einer Zeile
    for row_idx, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        ax = axes[row_idx]
        
        # Plotte alle Layer in diesem Subplot
        for layer_idx, layer_name in enumerate(layer_names):
            epochs_data = layer_data[layer_name]
            epochs = sorted(epochs_data.keys())
            
            # Extrahiere Metrik-Werte für diesen Layer
            metric_values = [epochs_data[ep][metric_key] for ep in epochs]
            
            # Wähle Farbe und Marker für diesen Layer
            color = layer_colors[layer_idx]
            marker = layer_markers[layer_idx % len(layer_markers)]
            
            # Plot der Layer-Daten
            ax.plot(
                epochs, metric_values,
                marker=marker, linestyle='-', linewidth=2,
                markersize=7, color=color, label=layer_name,
                alpha=0.8
            )
        
        # Baseline-Linie (gestrichelt) falls vorhanden
        if input_data_metrics is not None and metric_key in input_data_metrics:
            baseline_value = input_data_metrics[metric_key]
            ax.axhline(
                y=baseline_value,
                color=baseline_color, linestyle='--', linewidth=2,
                alpha=0.7, label='Input Data (Baseline)'
            )
        
        # Labels und Titel
        ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Epoche', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xticks(all_epochs)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=min(n_layers + (1 if input_data_metrics else 0), 5))
    
    plt.tight_layout()
    
    # Speichere Plot falls gewünscht
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = "manifold_metrics_all_layers_in_one_plot.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Plot gespeichert: {filepath}")
    
    return fig


def plot_manifold_metrics_over_layer(
    results: Union[List[Dict[str, Union[float, np.ndarray]]], Dict[tuple, Dict[str, Union[float, np.ndarray]]]],
    activity_logs_path: Optional[Union[str, List[str]]] = None,
    input_data_metrics: Optional[Dict[str, float]] = None,
    save_dir: Optional[str] = None,
    figsize_per_subplot: tuple = (6, 4)
) -> plt.Figure:
    """
    Erstellt einen Plot für Manifold-Metriken (Capacity, Radius, Dimension) über Layer.
    
    Der Plot zeigt Layer auf der X-Achse und Metrik-Werte auf der Y-Achse. Jede Epoche
    wird als separate Linie dargestellt. Jeder Subplot zeigt eine Metrik (Capacity, Radius, Dimension).
    Optional können Baseline-Werte aus Input-Daten als gestrichelte Linien angezeigt werden.
    
    Args:
        results: Entweder:
                1. Dictionary mit (epoch, layer_name) Tupeln als Keys und Ergebnis-Dictionaries als Values
                   Format: {(epoch, layer_name): {'capacity': float, 'radius': float, 'dimension': float}, ...}
                2. Liste von Ergebnis-Dictionaries (für Rückwärtskompatibilität)
        activity_logs_path: Optional, entweder:
                            - Pfad zum Verzeichnis mit Activity-Log-Dateien (wenn results eine Liste ist)
                            - Liste von Dateinamen (wenn results eine Liste ist)
                            - Wird ignoriert, wenn results ein Dictionary ist
        input_data_metrics: Optional, Dictionary mit Baseline-Metriken für Input-Daten.
                          Format: {'capacity': float, 'radius': float, 'dimension': float}
                          Wird als gestrichelte rote Linie in jedem Plot angezeigt.
        save_dir: Optional, Verzeichnis zum Speichern des Plots. Wenn None, wird Plot nicht gespeichert.
        figsize_per_subplot: Größe pro Subplot (default: (6, 4))
    
    Returns:
        matplotlib.pyplot.Figure: Die erstellte Figure
    """
    layer_data = {}  # {layer_name: {epoch: {metric: value}}}
    
    # Prüfe, ob results ein Dictionary oder eine Liste ist
    if isinstance(results, dict):
        # Neue flexible API: Dictionary mit (epoch, layer_name) -> result
        for (epoch, layer_name), result in results.items():
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            layer_data[layer_name][epoch] = {
                'capacity': result['capacity'],
                'radius': result['radius'],
                'dimension': result['dimension']
            }
    elif isinstance(results, list):
        # Alte API: Liste von Ergebnissen + activity_logs_path
        if activity_logs_path is None:
            raise ValueError("Wenn results eine Liste ist, muss activity_logs_path angegeben werden")
        
        # Prüfe, ob activity_logs_path ein Verzeichnis oder eine Liste von Dateinamen ist
        if isinstance(activity_logs_path, list):
            # Liste von Dateinamen
            file_names = activity_logs_path
        else:
            # Pfad zum Verzeichnis
            activity_logs_path = Path(activity_logs_path)
            if not activity_logs_path.exists():
                raise ValueError(f"Activity logs Pfad existiert nicht: {activity_logs_path}")
            h5_files = sorted(activity_logs_path.glob("*.h5"))
            file_names = [f.name for f in h5_files]
        
        if len(file_names) != len(results):
            raise ValueError(
                f"Anzahl der Dateien ({len(file_names)}) stimmt nicht mit Anzahl der Ergebnisse ({len(results)}) überein"
            )
        
        # Regex-Pattern für Dateinamen: epoch_XXX_layername_spk_events.h5
        pattern = re.compile(r'epoch_(\d+)_(\w+)_spk_events\.h5')
        
        for file_name, result in zip(file_names, results):
            match = pattern.match(file_name)
            if not match:
                print(f"⚠️ Warnung: Dateiname passt nicht zum erwarteten Format: {file_name}")
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
    else:
        raise TypeError(f"results muss ein Dictionary oder eine Liste sein, nicht {type(results)}")
    
    if not layer_data:
        raise ValueError("Keine gültigen Layer-Daten gefunden. Überprüfe die Dateinamen.")
    
    # Sortiere Layer für konsistente Reihenfolge
    layer_names = sorted(layer_data.keys())
    n_layers = len(layer_names)
    
    # Sammle alle Epochen für alle Layer
    all_epochs = set()
    for layer_name in layer_names:
        all_epochs.update(layer_data[layer_name].keys())
    all_epochs = sorted(all_epochs)
    
    # Erstelle ein Subplot-Grid: 3 Zeilen × 1 Spalte (eine Metrik pro Zeile)
    fig, axes = plt.subplots(
        3, 1, 
        figsize=(figsize_per_subplot[0], figsize_per_subplot[1] * 3)
    )
    
    # Falls nur eine Metrik, mache axes zu Array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    fig.suptitle('Manifold Metriken über Layer (Alle Epochen)', fontsize=16, fontweight='bold', y=0.995)
    
    # Metriken-Namen
    metric_names = ['Capacity (α_M)', 'Radius (R_M)', 'Dimension (D_M)']
    metric_keys = ['capacity', 'radius', 'dimension']
    baseline_color = 'red'
    
    # Farben für verschiedene Epochen
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(all_epochs)))
    epoch_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Layer-Positionen für X-Achse (numerisch für Plot, dann Labels)
    layer_positions = np.arange(len(layer_names))
    
    # Plotte jede Metrik in einer Zeile
    for row_idx, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        ax = axes[row_idx]
        
        # Plotte jede Epoche als separate Linie
        for epoch_idx, epoch in enumerate(all_epochs):
            # Sammle Metrik-Werte für diese Epoche über alle Layer
            metric_values = []
            for layer_name in layer_names:
                if epoch in layer_data[layer_name]:
                    metric_values.append(layer_data[layer_name][epoch][metric_key])
                else:
                    # Wenn Daten für diese Epoche fehlen, verwende NaN
                    metric_values.append(np.nan)
            
            # Wähle Farbe und Marker für diese Epoche
            color = epoch_colors[epoch_idx]
            marker = epoch_markers[epoch_idx % len(epoch_markers)]
            
            # Plot der Epochen-Daten über Layer
            ax.plot(
                layer_positions, metric_values,
                marker=marker, linestyle='-', linewidth=2,
                markersize=7, color=color, label=f'Epoch {epoch}',
                alpha=0.8
            )
        
        # Baseline-Linie (gestrichelt) falls vorhanden
        if input_data_metrics is not None and metric_key in input_data_metrics:
            baseline_value = input_data_metrics[metric_key]
            ax.axhline(
                y=baseline_value,
                color=baseline_color, linestyle='--', linewidth=2,
                alpha=0.7, label='Input Data (Baseline)'
            )
        
        # Labels und Titel
        ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xticks(layer_positions)
        ax.set_xticklabels(layer_names, rotation=0, ha='center')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=min(len(all_epochs) + (1 if input_data_metrics else 0), 5))
    
    plt.tight_layout()
    
    # Speichere Plot falls gewünscht
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = "manifold_metrics_over_layers_all_epochs.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Plot gespeichert: {filepath}")
    
    return fig


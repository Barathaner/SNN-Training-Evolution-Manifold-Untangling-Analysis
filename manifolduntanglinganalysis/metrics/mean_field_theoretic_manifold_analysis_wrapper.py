"""
Wrapper für Mean-Field-Theoretic Manifold Analysis.

Einfache Funktion zur Analyse von Manifold-Metriken aus einem DataLoader.
"""

import numpy as np
from typing import List, Dict, Union

from mftma.manifold_analysis_correlation import manifold_analysis_corr


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


import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import skdim


def _collect_data_from_dataloader(dataloader: DataLoader) -> np.ndarray:
    """
    Sammelt alle Daten aus einem DataLoader und formatiert sie für intrinsische Dimensions-Analyse.
    
    Für zeitabhängige Daten: Jeder Timestep × Sample = ein Datenpunkt
    Shape: [Batch, T, Features] -> [Batch*T, Features]
    
    Args:
        dataloader: DataLoader mit den Daten
    
    Returns:
        X: Array mit Shape [N_samples, Features] wobei N_samples = Batch*T über alle Batches
    """
    all_data = []
    
    for events, _ in dataloader:
        if events.ndim == 4:
            events = events.squeeze(2)
        
        events_np = events.numpy() if isinstance(events, torch.Tensor) else events
        batch_size, T, features = events_np.shape
        events_flat = events_np.reshape(batch_size * T, features)
        all_data.append(events_flat)
    
    X = np.concatenate(all_data, axis=0)
    return X


def explained_variance_dimension(dataloader: DataLoader, 
                                  plot_path: Optional[str] = None,
                                  perc: float = 0.90) -> int:
    """
    Berechnet die intrinsische Dimension mit PCA und erstellt einen Plot der kumulativen erklärten Varianz.
    
    Args:
        dataloader: DataLoader mit den Daten
        plot_path: Optional, Pfad zum Speichern des Plots. Wenn None, wird project_root/plots verwendet
        project_root: Optional, Projekt-Root-Verzeichnis für Standard-Speicherort
        perc: Prozentsatz der Varianz, der erklärt werden soll (default: 0.90)
    
    Returns:
        Anzahl der Dimensionen, die benötigt werden, um perc% der Varianz zu erklären
    """
    X = _collect_data_from_dataloader(dataloader)
    
    # Prüfe auf identische Datenpunkte
    unique_rows = np.unique(X, axis=0)
    if len(unique_rows) < len(X) * 0.95:  # Mehr als 5% Duplikate
        X = unique_rows
    # PCA durchführen
    pca = PCA()
    pca.fit(X)
    
    # Kumulative erklärte Varianz berechnen
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Finde die Anzahl der Dimensionen für perc% Varianz
    n_dims = np.argmax(cumsum_variance >= perc) + 1
    
    # Plot erstellen
    n_components = len(cumsum_variance)
    dimensions = np.arange(0, n_components)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dimensions, cumsum_variance, 'b-', linewidth=2, label='Kumulative erklärte Varianz')
    ax.axhline(y=perc, color='r', linestyle='--', linewidth=1.5, label=f'{perc*100:.0f}% Varianz')
    ax.axvline(x=n_dims-1, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Anzahl Dimensionen (Neuronen)', fontsize=12)
    ax.set_ylabel('Kumulative erklärte Varianz', fontsize=12)
    ax.set_title(f'PCA: Intrinsische Dimension der Neuronen-Aktivität\n'
                 f'(Jeder Timestep × Sample = Datenpunkt, benötigt {n_dims}/{X.shape[1]} Neuronen für {perc*100:.0f}% Varianz)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, n_components)
    ax.set_ylim(0, 1.0)
    
    # Speichere Plot
    if plot_path is None:
        plots_dir = Path.cwd() / 'plots'
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / 'explained_variance_intrinsic_dimension_pca.png'
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Intrinsische Dimension: {n_dims} Dimensionen erklären {perc*100:.0f}% der Varianz")
    print(f"✅ Plot gespeichert: {plot_path}")
    
    return n_dims,fig



def mle_intrinsic_dimension(dataloader: DataLoader,
                            normalize: bool = True,):
    """
    Schätzt die intrinsische Dimension mit Maximum Likelihood Estimation (MLE, Levina-Bickel).
    
    Gute Balance zwischen Genauigkeit und Geschwindigkeit.
    
    MLE kann NaN zurückgeben wenn:
    - Zu viele identische/ähnliche Datenpunkte (Distanzen = 0)
    - Numerische Probleme bei hohen Dimensionen
    - Zu wenige Datenpunkte
    
    Args:
        dataloader: DataLoader mit den Daten
        plot_path: Optional, Pfad zum Speichern des Plots
        project_root: Optional, Projekt-Root-Verzeichnis
        normalize: Ob Daten normalisiert werden sollen (empfohlen, default: True)
        max_samples: Maximale Anzahl Samples für MLE (default: 10000, None = alle)
    
    Returns:
        Geschätzte intrinsische Dimension
    """

    X = _collect_data_from_dataloader(dataloader)
    
    
    
    # Normalisierung (wichtig für numerische Stabilität)
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Prüfe auf identische Datenpunkte
    unique_rows = np.unique(X, axis=0)
    if len(unique_rows) < len(X) * 0.95:  # Mehr als 5% Duplikate
        X = unique_rows
    

    estimator = skdim.id.MLE()
    estimator.fit(X)
    dim = estimator.dimension_

    print(f"✅ MLE geschätzte Dimension: {dim:.2f}")
    return dim

def twonn_intrinsic_dimension(dataloader: DataLoader,
                             normalize: bool = True,
                             max_samples: int = 30000) -> float:
    """
    Schätzt die intrinsische Dimension mit Two-NN (Facco et al. 2017).
    
    Oft am robustesten für gekrümmte Manifolds.
    Args:
        dataloader: DataLoader mit den Daten
        normalize: Ob Daten normalisiert werden sollen (default: True)
    
    Returns:
        Geschätzte intrinsische Dimension
    """
    X = _collect_data_from_dataloader(dataloader)
    
    # Subsampling für große Datensätze (Two-NN ist O(n²) - sehr langsam!)
    if max_samples is not None and X.shape[0] > max_samples:
        print(f"⚠️ Zu viele Samples ({X.shape[0]}), subsample auf {max_samples} für Two-NN")
        print(f"   (Two-NN ist O(n²) - würde sonst sehr lange dauern)")
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
    
    # Normalisierung (wichtig für numerische Stabilität)
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Prüfe auf identische Datenpunkte
    unique_rows = np.unique(X, axis=0)
    if len(unique_rows) < len(X) * 0.95:  # Mehr als 5% Duplikate
        X = unique_rows
    estimator = skdim.id.TwoNN()
    estimator.fit(X)
    dim = estimator.dimension_
    
    print(f"✅ Two-NN geschätzte Dimension: {dim:.2f}")
    return dim



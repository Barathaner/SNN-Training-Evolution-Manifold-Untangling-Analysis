import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ripser import ripser
from persim import plot_diagrams
from torch.utils.data import DataLoader
import torch


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


def compute_persistence_diagrams(dataloader: DataLoader, maxdim=2):
    """
    Berechnet Persistence Diagrams für gegebene Embeddings.
    
    WICHTIG: ripser ist O(n³) und sehr speicherintensiv. Bei >2000 Punkten kann es zu 
    Memory-Fehlern kommen. Verwende max_points für Subsampling.
    
    Args:
        embeddings: Array mit Shape (n_points, n_features)
        maxdim: Maximale Homologie-Dimension (default: 2)
        max_points: Maximale Anzahl Punkte (default: 1500). Wenn None, werden alle verwendet.
                   Empfohlen: 1000-1500 für stabile Berechnung
    
    Returns:
        Persistence Diagrams (ripser output)
    """

    embeddings = _collect_data_from_dataloader(dataloader)
    # Entferne Duplikate
    embeddings = np.unique(embeddings, axis=0)
    
    print(f"Maximale Dimension: H{maxdim}")
    
    # Optimierte ripser Parameter für bessere Performance
    diagrams = ripser(
        embeddings, 
        maxdim=maxdim,
    )
    print("✅ Persistence Diagrams erfolgreich berechnet!")
    return diagrams


def plot_persistence_diagrams(diagrams, ax=None, show=True):
    """
    Plottet Persistence Diagrams.
    
    Args:
        diagrams: Persistence Diagrams (von ripser)
        ax: Optional, matplotlib Axes (wenn None, wird neuer Plot erstellt)
        show: Ob plt.show() aufgerufen werden soll (default: True)
    
    Returns:
        matplotlib Axes
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    
    plot_diagrams(diagrams['dgms'], ax=ax, show=False)
    
    if created_fig and show:
        plt.show()
    
    return ax


def plot_betti_barcodes(diagrams, ax=None, show=True):
    """
    Plottet Betti Barcodes aus Persistence Diagrams.
    
    Args:
        diagrams: Persistence Diagrams (von ripser)
        ax: Optional, matplotlib Axes (wenn None, wird neuer Plot erstellt)
        show: Ob plt.show() aufgerufen werden soll (default: True)
    
    Returns:
        matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    dgms = diagrams['dgms']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    y_offset = 0
    for dim in range(len(dgms)):
        if len(dgms[dim]) == 0:
            continue
        
        bars = dgms[dim]
        for i, (birth, death) in enumerate(bars):
            if np.isinf(death):
                death = birth + 1  # Für unendliche Features: zeige bis birth+1
            
            ax.plot([birth, death], [y_offset + i, y_offset + i], 
                   color=colors[dim % len(colors)], linewidth=2, label=f'H{dim}' if i == 0 else '')
        
        y_offset += len(bars) + 1
    
    ax.set_xlabel('Filtrationsparameter (ε)', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title('Betti Barcodes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if show:
        plt.show()
    
    return ax


def compute_and_plot_persistence(embeddings, 
                                  maxdim=2,
                                  max_points: Optional[int] = 1500,
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> Tuple[dict, plt.Figure]:
    """
    Berechnet Persistence Diagrams und erstellt einen kombinierten Plot mit
    Persistence Diagrams und Betti Barcodes.
    
    Args:
        embeddings: Array mit Shape (n_points, n_features)
        maxdim: Maximale Homologie-Dimension (default: 2)
        max_points: Maximale Anzahl Punkte für ripser (default: 1500)
                   Empfohlen: 1000-1500 für stabile Berechnung ohne Memory-Fehler
        save_path: Optional, Pfad zum Speichern des Plots
        show: Ob plt.show() aufgerufen werden soll (default: True)
    
    Returns:
        Tuple mit (diagrams, figure)
    """
    diagrams = compute_persistence_diagrams(embeddings, maxdim=maxdim, max_points=max_points)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Persistent Homology Analysis', fontsize=16, fontweight='bold')
    
    # Persistence Diagrams
    ax1 = axes[0]
    plot_persistence_diagrams(diagrams, ax=ax1, show=False)
    ax1.set_title('Persistence Diagrams', fontsize=12, fontweight='bold')
    
    # Betti Barcodes
    ax2 = axes[1]
    plot_betti_barcodes(diagrams, ax=ax2, show=False)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot gespeichert: {save_path}")
    
    if show:
        plt.show()
    
    return diagrams, fig

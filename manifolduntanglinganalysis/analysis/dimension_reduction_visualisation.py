"""
Builder-Klasse für vergleichende Dimensionsreduktions-Visualisierungen.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Union
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap


def _collect_trajectories_from_dataloader(dataloader: DataLoader, max_samples: Optional[int] = None) -> tuple:
    all_data = []
    all_labels = []
    
    for events, labels in dataloader:
        if events.ndim == 4:
            events = events.squeeze(2)
        
        events_np = events.numpy() if isinstance(events, torch.Tensor) else events
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        batch_size, T, features = events_np.shape
        # Jeder Timestep × Sample = ein Datenpunkt: [Batch*T, Features]
        events_flat = events_np.reshape(batch_size * T, features)
        # Labels für jeden Timestep (gleiches Label für alle Timesteps eines Samples)
        labels_flat = np.repeat(labels_np, T)
        
        all_data.append(events_flat)
        all_labels.append(labels_flat)
    
    X = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if max_samples is not None and X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        labels = labels[indices]
    
    return X, labels


class DimensionReductionVisualizer:
    def __init__(self, data: Union[DataLoader, np.ndarray], 
                 labels: Optional[np.ndarray] = None,
                 max_samples: Optional[int] = None):
        if isinstance(data, DataLoader):
            X, labels_from_dl = _collect_trajectories_from_dataloader(data, max_samples)
            labels = labels if labels is not None else labels_from_dl
        else:
            X = data
            if X.ndim == 3:
                # Jeder Timestep × Sample = ein Datenpunkt: [Batch*T, Features]
                batch_size, T, features = X.shape
                X = X.reshape(batch_size * T, features)
                # Labels für jeden Timestep wiederholen
                if labels is not None and labels.ndim == 1 and len(labels) == batch_size:
                    labels = np.repeat(labels, T)
        
        self.X = X
        self.labels = labels
        self.reductions = {}
        
    def add_pca(self, name: str = "PCA", n_components: int = 2, **kwargs):
        """Fügt PCA hinzu."""
        reducer = PCA(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(self.X)
        self.reductions[name] = (reducer, embedding)
        return self
    
    def add_tsne(self, name: str = "t-SNE", n_components: int = 2, 
                 perplexity: float = 30.0, random_state: int = 42, **kwargs):
        """Fügt t-SNE hinzu."""
        n_samples = self.X.shape[0]
        
        # t-SNE erfordert: perplexity < n_samples
        # Empfohlen: perplexity sollte zwischen 5 und 50 liegen, aber < n_samples
        if perplexity >= n_samples:
            adjusted_perplexity = max(5, min(50, n_samples - 1))
            print(f"⚠️ Warnung: Perplexity ({perplexity}) >= Anzahl Samples ({n_samples})")
            print(f"   Automatisch angepasst auf {adjusted_perplexity}")
            perplexity = adjusted_perplexity
        elif perplexity > 50:
            print(f"⚠️ Warnung: Perplexity ({perplexity}) > 50, kann zu schlechteren Ergebnissen führen")
        
        reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                      random_state=random_state, **kwargs)
        embedding = reducer.fit_transform(self.X)
        
        # Prüfe auf NaN oder extreme Werte
        if np.any(np.isnan(embedding)):
            print(f"⚠️ Warnung: t-SNE hat NaN-Werte erzeugt (perplexity={perplexity}, n_samples={n_samples})")
            print(f"   Versuche niedrigere Perplexity oder mehr Samples")
        elif np.any(np.abs(embedding) > 1e6):
            print(f"⚠️ Warnung: t-SNE hat extreme Werte erzeugt (perplexity={perplexity})")
        
        self.reductions[name] = (reducer, embedding)
        return self
    
    def add_umap(self, name: str = "UMAP", n_components: int = 2,
                 n_neighbors: int = 15, min_dist: float = 0.1, 
                 random_state: int = 42, **kwargs):
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=random_state, 
            low_memory=True,**kwargs)
        embedding = reducer.fit_transform(self.X)
        self.reductions[name] = (reducer, embedding)
        return self
    
    def add_isomap(self, name: str = "Isomap", n_components: int = 2,
                   n_neighbors: int = 5, **kwargs):
        """Fügt Isomap hinzu."""
        reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
        embedding = reducer.fit_transform(self.X)
        self.reductions[name] = (reducer, embedding)
        return self
    
    def add_custom(self, name: str, reducer, **fit_kwargs):
        embedding = reducer.fit_transform(self.X, **fit_kwargs)
        self.reductions[name] = (reducer, embedding)
        return self
    
    def build_plot(self, 
                   figsize: Optional[tuple] = None,
                   save_path: Optional[str] = None,
                   title: str = "Dimensionsreduktions-Vergleich",
                   show_legend: bool = True,
                   colormap: str = "tab10") -> plt.Figure:
        if not self.reductions:
            raise ValueError("Keine Reduktionen hinzugefügt. Verwende add_pca(), add_umap(), etc.")
        
        n_methods = len(self.reductions)
        figsize = figsize or (5 * n_methods, 5)
        
        fig, axes = plt.subplots(1, n_methods, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for idx, (name, (_, embedding)) in enumerate(self.reductions.items()):
            ax = axes[idx]
            
            # Prüfe auf NaN oder extreme Werte
            valid_mask = ~np.any(np.isnan(embedding), axis=1) & ~np.any(np.isinf(embedding), axis=1)
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                print(f"⚠️ {name}: {n_invalid} ungültige Punkte (NaN/Inf) werden übersprungen")
                embedding = embedding[valid_mask]
                if self.labels is not None:
                    labels_plot = self.labels[valid_mask]
                else:
                    labels_plot = None
            else:
                labels_plot = self.labels
            
            # Prüfe auf extreme Werte (außerhalb eines vernünftigen Bereichs)
            if embedding.shape[1] == 2:
                x_range = np.ptp(embedding[:, 0])
                y_range = np.ptp(embedding[:, 1])
                if x_range > 1e6 or y_range > 1e6:
                    print(f"⚠️ {name}: Extreme Werte erkannt (Range: x={x_range:.2e}, y={y_range:.2e})")
                    # Filtere extreme Werte für bessere Visualisierung
                    x_valid = np.abs(embedding[:, 0]) < 1e6
                    y_valid = np.abs(embedding[:, 1]) < 1e6
                    valid_mask = x_valid & y_valid
                    if not np.all(valid_mask):
                        embedding = embedding[valid_mask]
                        if labels_plot is not None:
                            labels_plot = labels_plot[valid_mask]
            
            if embedding.shape[1] == 3:
                ax = fig.add_subplot(1, n_methods, idx + 1, projection='3d')
                scatter_kwargs = {'c': labels_plot, 'cmap': colormap} if labels_plot is not None else {}
                ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.6, s=20, **scatter_kwargs)
            else:
                scatter_kwargs = {'c': labels_plot, 'cmap': colormap} if labels_plot is not None else {}
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20,
                                   edgecolors='k', linewidths=0.5, **scatter_kwargs)
                if show_legend and labels_plot is not None and idx == 0:
                    plt.colorbar(scatter, ax=ax, label='Label')
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Komponente 1')
            ax.set_ylabel('Komponente 2')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot gespeichert: {save_path}")
        
        return fig
    
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        return {name: embedding for name, (_, embedding) in self.reductions.items()}


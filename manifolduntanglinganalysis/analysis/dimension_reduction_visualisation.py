"""
Builder-Klasse für vergleichende Dimensionsreduktions-Visualisierungen.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap

try:
    from manifolduntanglinganalysis.preprocessing.metadata_extractor import extract_batch_metadata
except ImportError:
    extract_batch_metadata = None


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


def _collect_trajectories_with_speakers_from_dataloader(dataloader: DataLoader, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sammelt Trajektorien-Daten mit Speaker-IDs aus einem SHD DataLoader.
    
    Args:
        dataloader: DataLoader mit SHD-Daten
        max_samples: Maximale Anzahl Datenpunkte (optional)
    
    Returns:
        Tuple mit (X, labels, speakers) wo:
        - X: Array mit Shape [N_samples, Features]
        - labels: Array mit Labels für jeden Datenpunkt
        - speakers: Array mit Speaker-IDs für jeden Datenpunkt
    """
    if extract_batch_metadata is None:
        raise ImportError("extract_batch_metadata nicht verfügbar. Stelle sicher, dass metadata_extractor importierbar ist.")
    
    all_data = []
    all_labels = []
    all_speakers = []
    
    batch_idx = 0
    for events, labels in dataloader:
        if events.ndim == 4:
            events = events.squeeze(2)
        
        events_np = events.numpy() if isinstance(events, torch.Tensor) else events
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        # Extrahiere Speaker-IDs für diesen Batch
        try:
            metadata = extract_batch_metadata(dataloader, batch_idx)
            batch_speakers = metadata['speakers']
        except (AttributeError, KeyError, ValueError) as e:
            raise ValueError(f"Konnte Speaker-IDs nicht extrahieren. Ist dies ein SHD-Dataset? Fehler: {e}")
        
        batch_size, T, features = events_np.shape
        # Jeder Timestep × Sample = ein Datenpunkt: [Batch*T, Features]
        events_flat = events_np.reshape(batch_size * T, features)
        # Labels für jeden Timestep (gleiches Label für alle Timesteps eines Samples)
        labels_flat = np.repeat(labels_np, T)
        # Speaker-IDs für jeden Timestep (gleicher Speaker für alle Timesteps eines Samples)
        speakers_flat = np.repeat(batch_speakers, T)
        
        all_data.append(events_flat)
        all_labels.append(labels_flat)
        all_speakers.append(speakers_flat)
        
        batch_idx += 1
    
    X = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    speakers = np.concatenate(all_speakers, axis=0)
    
    if max_samples is not None and X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        labels = labels[indices]
        speakers = speakers[indices]
    
    return X, labels, speakers


class DimensionReductionVisualizer:
    def __init__(self, data: Union[DataLoader, np.ndarray], 
                 labels: Optional[np.ndarray] = None,
                 max_samples: Optional[int] = None):
        self.speakers = None
        self._dataloader = None
        
        if isinstance(data, DataLoader):
            X, labels_from_dl = _collect_trajectories_from_dataloader(data, max_samples)
            labels = labels if labels is not None else labels_from_dl
            # Speichere DataLoader für später (falls Speaker-IDs benötigt werden)
            self._dataloader = data
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
        
        # Berechne optimale quadratische Anordnung
        n_cols = int(np.ceil(np.sqrt(n_methods)))
        n_rows = int(np.ceil(n_methods / n_cols))
        
        # Figsize anpassen für quadratisches Grid
        plot_size = 5
        if figsize is None:
            figsize = (plot_size * n_cols, plot_size * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
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
                ax.remove()
                ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
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
        
        # Verstecke leere Subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
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
    
    def build_plot_speaker_shd(self,
                               dataloader: Optional[DataLoader] = None,
                               figsize: Optional[tuple] = None,
                               save_path: Optional[str] = None,
                               title: str = "Dimensionsreduktions-Vergleich (nach Speaker)",
                               show_legend: bool = True,
                               colormap: str = "tab20") -> plt.Figure:
        """
        Erstellt einen Plot mit N×N Grid, wobei die Farbcodierung nach Speaker-IDs erfolgt.
        Nur für SHD-Datasets geeignet.
        
        Args:
            dataloader: Optional, DataLoader mit SHD-Daten. Wenn None, wird der beim 
                       Initialisieren verwendete DataLoader verwendet.
            figsize: Optional, Größe der Figur
            save_path: Optional, Pfad zum Speichern des Plots
            title: Titel des Plots
            show_legend: Ob Colorbar angezeigt werden soll
            colormap: Colormap für Speaker-Farbcodierung (default: 'tab20' für mehr Speaker)
        
        Returns:
            matplotlib Figure
        """
        if not self.reductions:
            raise ValueError("Keine Reduktionen hinzugefügt. Verwende add_pca(), add_umap(), etc.")
        
        # Verwende übergebenen DataLoader oder den beim Initialisieren gespeicherten
        if dataloader is None:
            if self._dataloader is None:
                raise ValueError("Kein DataLoader verfügbar. Entweder beim Initialisieren einen DataLoader "
                               "verwenden oder einen als Parameter übergeben.")
            dataloader = self._dataloader
        
        # Sammle Speaker-IDs (mit gleicher max_samples Logik wie beim Initialisieren)
        try:
            # Verwende die gleiche max_samples wie beim Initialisieren (falls vorhanden)
            # Da wir die ursprünglichen Indizes nicht kennen, sammeln wir alle und prüfen dann
            X_speaker, _, speakers = _collect_trajectories_with_speakers_from_dataloader(dataloader)
        except Exception as e:
            raise ValueError(f"Konnte Speaker-IDs nicht extrahieren: {e}. "
                           "Stelle sicher, dass der DataLoader ein SHD-Dataset enthält.")
        
        # Prüfe, ob die Daten übereinstimmen
        if X_speaker.shape[0] != self.X.shape[0]:
            # Wenn unterschiedlich, versuche die ersten N zu nehmen (falls keine Subsampling verwendet wurde)
            if X_speaker.shape[0] > self.X.shape[0]:
                speakers = speakers[:self.X.shape[0]]
                print(f"⚠️ Warnung: Speaker-Daten ({X_speaker.shape[0]}) > Reduktionen ({self.X.shape[0]}). "
                      "Verwende die ersten {self.X.shape[0]} Speaker-IDs.")
            else:
                raise ValueError(f"Anzahl der Datenpunkte stimmt nicht überein: "
                               f"Speaker-Daten: {X_speaker.shape[0]}, Reduktionen: {self.X.shape[0]}. "
                               "Verwende denselben DataLoader wie bei der Initialisierung.")
        
        n_methods = len(self.reductions)
        
        # Berechne optimale quadratische Anordnung
        n_cols = int(np.ceil(np.sqrt(n_methods)))
        n_rows = int(np.ceil(n_methods / n_cols))
        
        # Figsize anpassen für quadratisches Grid
        plot_size = 5
        if figsize is None:
            figsize = (plot_size * n_cols, plot_size * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for idx, (name, (_, embedding)) in enumerate(self.reductions.items()):
            ax = axes[idx]
            
            # Prüfe auf NaN oder extreme Werte
            valid_mask = ~np.any(np.isnan(embedding), axis=1) & ~np.any(np.isinf(embedding), axis=1)
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                print(f"⚠️ {name}: {n_invalid} ungültige Punkte (NaN/Inf) werden übersprungen")
                embedding = embedding[valid_mask]
                speakers_plot = speakers[valid_mask]
            else:
                speakers_plot = speakers
            
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
                        speakers_plot = speakers_plot[valid_mask]
            
            if embedding.shape[1] == 3:
                ax.remove()
                ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
                scatter_kwargs = {'c': speakers_plot, 'cmap': colormap}
                ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.6, s=20, **scatter_kwargs)
            else:
                scatter_kwargs = {'c': speakers_plot, 'cmap': colormap}
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20,
                                   edgecolors='k', linewidths=0.5, **scatter_kwargs)
                if show_legend and idx == 0:
                    plt.colorbar(scatter, ax=ax, label='Speaker ID')
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Komponente 1')
            ax.set_ylabel('Komponente 2')
            ax.grid(True, alpha=0.3)
        
        # Verstecke leere Subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot gespeichert: {save_path}")
        
        return fig


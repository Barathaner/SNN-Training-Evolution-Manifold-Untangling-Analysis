import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
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


def plot_intrinsic_dimensions_over_layers(
    results: dict,
    input_data_metrics: Optional[dict] = None,
    save_dir: Optional[str] = None,
    figsize_per_subplot: tuple = (6, 4),
    results_json_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Erstellt Plots für intrinsische Dimensionen (PCA, MLE, Two-NN) über Layer.
    
    Der Plot zeigt Layer auf der X-Achse und intrinsische Dimension auf der Y-Achse. 
    Jede Epoche wird als separate Linie dargestellt. Es werden 3 Subplots erstellt:
    einer für PCA, einer für MLE und einer für Two-NN.
    
    Args:
        results: Dictionary mit Struktur {epoch: {layer: {'PCA': float, 'MLE': float, 'Two-NN': float}}}
        input_data_metrics: Optional, Dictionary mit Baseline-Werten für Input-Daten.
                          Format: {'PCA': float, 'MLE': float, 'Two-NN': float}
                          Wird als gestrichelte rote Linie in jedem Plot angezeigt.
        save_dir: Optional, Verzeichnis zum Speichern des Plots. Wenn None, wird Plot nicht gespeichert.
        figsize_per_subplot: Größe pro Subplot (default: (6, 4))
        results_json_path: Optional, Pfad zu einer JSON-Datei mit Ergebnissen.
                          Format: {"epoch": {"layer": {"PCA": ..., "MLE": ..., "Two-NN": ...}}}
                          Wenn gesetzt, werden die Daten aus der JSON-Datei geladen und results ignoriert.
    
    Returns:
        matplotlib.pyplot.Figure: Die erstellte Figure
    """
    import json
    from pathlib import Path
    
    # Lade Daten aus JSON, falls results_json_path angegeben ist
    if results_json_path is not None:
        results_json_path = Path(results_json_path)
        if not results_json_path.exists():
            raise ValueError(f"JSON-Datei existiert nicht: {results_json_path}")
        
        with open(results_json_path, 'r') as f:
            results = json.load(f)
    
    if results is None or not isinstance(results, dict):
        raise ValueError("results muss ein Dictionary sein oder results_json_path muss angegeben werden")
    
    # Strukturiere Daten: {layer_name: {epoch: {metric: value}}}
    layer_data = {}
    
    for epoch, epoch_data in results.items():
        epoch = int(epoch)  # Konvertiere zu int für Sortierung
        if not isinstance(epoch_data, dict):
            continue
        
        for layer_name, layer_metrics in epoch_data.items():
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            layer_data[layer_name][epoch] = {
                'PCA': float(layer_metrics.get('PCA', 0.0)),
                'MLE': float(layer_metrics.get('MLE', 0.0)),
                'Two-NN': float(layer_metrics.get('Two-NN', 0.0))
            }
    
    if not layer_data:
        raise ValueError("Keine gültigen Layer-Daten gefunden.")
    
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
    
    fig.suptitle('Intrinsische Dimensionen über Layer (Alle Epochen)', fontsize=16, fontweight='bold', y=0.995)
    
    # Metriken-Namen
    metric_names = ['PCA (80% Varianz)', 'MLE (Maximum Likelihood)', 'Two-NN']
    metric_keys = ['PCA', 'MLE', 'Two-NN']
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
                    metric_values.append(layer_data[layer_name][epoch].get(metric_key, 0.0))
                else:
                    metric_values.append(np.nan)  # Fehlende Werte als NaN
            
            # Wähle Farbe und Marker für diese Epoche
            color = epoch_colors[epoch_idx]
            marker = epoch_markers[epoch_idx % len(epoch_markers)]
            
            # Plot der Epoche-Daten
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
        ax.set_ylabel('Intrinsische Dimension', fontsize=11)
        ax.set_xticks(layer_positions)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=min(len(all_epochs) + (1 if input_data_metrics else 0), 5))
    
    plt.tight_layout()
    
    # Speichere Plot falls gewünscht
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / 'intrinsic_dimensions_over_layers.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Plot gespeichert: {filepath}")
    
    return fig



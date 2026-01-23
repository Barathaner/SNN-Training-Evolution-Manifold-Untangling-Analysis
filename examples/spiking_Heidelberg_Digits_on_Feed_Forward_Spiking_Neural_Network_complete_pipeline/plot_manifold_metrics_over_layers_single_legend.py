"""
Skript zum Erstellen von Manifold-Metriken über Layer Plots mit nur einer Legende.
Liest results.json und erstellt den Plot mit allen Epochen, aber nur einer gemeinsamen Legende.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


def load_results_from_json(json_path: Path) -> Dict:
    """
    Lädt Ergebnisse aus JSON-Datei.
    
    Args:
        json_path: Pfad zur JSON-Datei mit Struktur: {"epoch": {"layer": {"capacity": ..., "radius": ..., "dimension": ..., "correlation": ...}}}
    
    Returns:
        Dictionary mit (epoch, layer_name) Tupeln als Keys: {(epoch, layer_name): {'capacity': float, 'radius': float, 'dimension': float, 'correlation': float}, ...}
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Konvertiere von {"epoch": {"layer": {...}}} zu {(epoch, layer): {...}}
    results = {}
    for epoch_str, layers in data.items():
        epoch = int(epoch_str)
        for layer_name, metrics in layers.items():
            results[(epoch, layer_name)] = {
                'capacity': float(metrics['capacity']),
                'radius': float(metrics['radius']),
                'dimension': float(metrics['dimension']),
                'correlation': float(metrics.get('correlation', 0.0))
            }
    
    return results


def plot_manifold_metrics_over_layers_single_legend(
    results_json_path: str,
    input_data_metrics: Optional[Dict[str, float]] = None,
    save_dir: Optional[str] = None,
    figsize_per_subplot: tuple = (6, 4)
) -> plt.Figure:
    """
    Erstellt einen Plot für Manifold-Metriken über Layer mit nur einer gemeinsamen Legende.
    
    Args:
        results_json_path: Pfad zu einer JSON-Datei mit Ergebnissen
        input_data_metrics: Optional, Dictionary mit Baseline-Metriken für Input-Daten
        save_dir: Optional, Verzeichnis zum Speichern des Plots
        figsize_per_subplot: Größe pro Subplot (default: (6, 4))
    
    Returns:
        matplotlib.pyplot.Figure: Die erstellte Figure
    """
    # Lade Daten aus JSON
    results = load_results_from_json(results_json_path)
    
    layer_data = {}  # {layer_name: {epoch: {metric: value}}}
    
    # Konvertiere zu layer_data Format
    for (epoch, layer_name), result in results.items():
        if layer_name not in layer_data:
            layer_data[layer_name] = {}
        
        layer_data[layer_name][epoch] = {
            'capacity': result['capacity'],
            'radius': result['radius'],
            'dimension': result['dimension'],
            'correlation': result.get('correlation', 0.0)
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
    
    # Erstelle ein Subplot-Grid: 2 Zeilen × 2 Spalten
    # Links: Capacity (oben), Radius (unten)
    # Rechts: Dimension (oben), Correlation (unten)
    fig, axes = plt.subplots(
        2, 2, 
        figsize=(figsize_per_subplot[0] * 2, figsize_per_subplot[1] * 2)
    )
    
    # Flatten axes für einfachere Indizierung
    axes = axes.flatten()
    
    fig.suptitle('Manifold Metriken über Layer (Alle Epochen)', fontsize=16, fontweight='bold', y=0.995)
    
    # Metriken-Namen und Reihenfolge für 2x2 Grid:
    # [0] = Capacity (links oben)
    # [1] = Dimension (rechts oben)
    # [2] = Radius (links unten)
    # [3] = Correlation (rechts unten)
    metric_names = ['Capacity (α_M)', 'Dimension (D_M)', 'Radius (R_M)', 'Correlation']
    metric_keys = ['capacity', 'dimension', 'radius', 'correlation']
    baseline_color = 'red'
    
    # Farben für verschiedene Epochen
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(all_epochs)))
    epoch_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Layer-Positionen für X-Achse
    layer_positions = np.arange(len(layer_names))
    
    # Sammle alle Handles für die gemeinsame Legende
    legend_handles = []
    legend_labels = []
    
    # Plotte jede Metrik in einem Subplot
    for plot_idx, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        ax = axes[plot_idx]
        
        # Plotte jede Epoche als separate Linie
        for epoch_idx, epoch in enumerate(all_epochs):
            # Sammle Metrik-Werte für diese Epoche über alle Layer
            metric_values = []
            for layer_name in layer_names:
                if epoch in layer_data[layer_name]:
                    metric_values.append(layer_data[layer_name][epoch].get(metric_key, 0.0))
                else:
                    metric_values.append(np.nan)
            
            # Wähle Farbe und Marker für diese Epoche
            color = epoch_colors[epoch_idx]
            marker = epoch_markers[epoch_idx % len(epoch_markers)]
            
            # Plot der Epochen-Daten über Layer
            line = ax.plot(
                layer_positions, metric_values,
                marker=marker, linestyle='-', linewidth=2,
                markersize=7, color=color, label=f'Epoch {epoch}',
                alpha=0.8
            )
            
            # Sammle Handle nur beim ersten Subplot (um Duplikate zu vermeiden)
            if plot_idx == 0:
                legend_handles.append(line[0])
                legend_labels.append(f'Epoch {epoch}')
        
        # Baseline-Linie (gestrichelt) falls vorhanden
        if input_data_metrics is not None and metric_key in input_data_metrics:
            baseline_value = input_data_metrics[metric_key]
            baseline_line = ax.axhline(
                y=baseline_value,
                color=baseline_color, linestyle='--', linewidth=2,
                alpha=0.7, label='Input Data (Baseline)'
            )
            
            # Sammle Baseline-Handle nur beim ersten Subplot
            if plot_idx == 0:
                legend_handles.append(baseline_line)
                legend_labels.append('Input Data (Baseline)')
        
        # Labels und Titel
        ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xticks(layer_positions)
        ax.set_xticklabels(layer_names, rotation=0, ha='center')
        ax.grid(True, alpha=0.3)
        # KEINE Legende in einzelnen Subplots
    
    # Füge gemeinsame Legende hinzu (außerhalb der Subplots)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower center',
        ncol=min(len(all_epochs) + (1 if input_data_metrics else 0), 10),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Speichere Plot falls gewünscht
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = "manifold_metrics_over_layers_all_epochs_single_legend.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Plot gespeichert: {filepath}")
    
    return fig


if __name__ == "__main__":
    import sys
    
    # Projekt-Root bestimmen
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    results_json_path = os.path.join(project_root, "data", "results", "results.json")
    plots_dir = os.path.join(project_root, "plots")
    
    # Lade input_data_metrics (optional)
    input_metrics_path = os.path.join(project_root, "data", "results", "input_data_metrics.json")
    input_data_metrics = None
    
    if os.path.exists(input_metrics_path):
        print(f"📂 Lade input_data_metrics aus {input_metrics_path}")
        with open(input_metrics_path, 'r') as f:
            input_data_metrics = json.load(f)
    else:
        print("⚠️  input_data_metrics.json nicht gefunden, verwende Standardwerte")
        input_data_metrics = {
            'capacity': 0.0069,
            'radius': 1.8510,
            'dimension': 187.0195,
            'correlation': 0.5949
        }
    
    print(f"{'='*80}")
    print("📊 Manifold Metriken über Layer - Plot Generator (Single Legend)")
    print(f"{'='*80}")
    print(f"📁 Results JSON: {results_json_path}")
    print(f"📁 Output: {plots_dir}")
    print(f"{'='*80}\n")
    
    try:
        plot_manifold_metrics_over_layers_single_legend(
            results_json_path=results_json_path,
            input_data_metrics=input_data_metrics,
            save_dir=plots_dir,
            figsize_per_subplot=(6, 4)
        )
        print("\n✅ Plot erfolgreich erstellt!")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

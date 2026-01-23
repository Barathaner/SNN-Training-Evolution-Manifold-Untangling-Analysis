"""
Skript zum Erstellen von Plots für PCA-Dimension über Epochen für jeden Layer.
Liest results.json und erstellt für jeden Layer einen Plot.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_pca_dimension_over_epochs(results_json_path, output_dir=None):
    """
    Erstellt Plots für PCA-Dimension über Epochen für jeden Layer.
    
    Args:
        results_json_path: Pfad zur results.json Datei
        output_dir: Verzeichnis zum Speichern der Plots (optional)
    """
    # Lade results.json
    results_json_path = Path(results_json_path)
    if not results_json_path.exists():
        raise FileNotFoundError(f"results.json nicht gefunden: {results_json_path}")
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    print(f"📂 Geladen: {results_json_path}")
    
    # Bestimme Output-Verzeichnis
    if output_dir is None:
        output_dir = results_json_path.parent.parent / "plots"
    else:
        output_dir = Path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sammle alle Layer-Namen
    all_layers = set()
    for epoch_data in results.values():
        all_layers.update(epoch_data.keys())
    
    all_layers = sorted(all_layers)
    print(f"📊 Gefundene Layer: {all_layers}\n")
    
    # Erstelle Plot für jeden Layer
    for layer_name in all_layers:
        print(f"📈 Erstelle Plot für {layer_name}...")
        
        # Sammle Daten für diesen Layer
        epochs = []
        pca_dimensions = []
        
        for epoch_str, epoch_data in sorted(results.items(), key=lambda x: int(x[0])):
            if layer_name in epoch_data:
                epoch = int(epoch_str)
                pca_dim = epoch_data[layer_name].get('pca_intdim')
                
                if pca_dim is not None:
                    epochs.append(epoch)
                    pca_dimensions.append(float(pca_dim))
        
        if len(epochs) == 0:
            print(f"   ⚠️  Keine Daten für {layer_name}")
            continue
        
        # Erstelle Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mit Linie und Markern
        ax.plot(epochs, pca_dimensions, marker='o', linestyle='-', linewidth=2, 
                markersize=8, label='PCA Dimension', color='#2ca02c')
        
        # Beschriftungen
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('PCA Intrinsic Dimension', fontsize=12, fontweight='bold')
        ax.set_title(f'PCA Intrinsic Dimension über Epochen - {layer_name}', 
                    fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # X-Achse: Ganze Zahlen für Epochs
        ax.set_xticks(epochs)
        ax.set_xticklabels(epochs)
        
        # Y-Achse: Ganze Zahlen wenn möglich
        y_min = int(np.floor(min(pca_dimensions)))
        y_max = int(np.ceil(max(pca_dimensions)))
        y_range = y_max - y_min
        if y_range <= 10:
            ax.set_yticks(range(y_min, y_max + 1))
        else:
            # Bei größerer Range: automatische Ticks
            ax.set_yticks(np.linspace(y_min, y_max, min(10, y_range + 1)))
        
        # Layout
        plt.tight_layout()
        
        # Speichere Plot
        output_path = output_dir / f"pca_dimension_{layer_name}_over_epochs.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Gespeichert: {output_path}")
        print(f"      Epochen: {len(epochs)}, Dimension Range: {min(pca_dimensions):.1f} - {max(pca_dimensions):.1f}\n")
    
    print("✅ Alle Plots erstellt!")


if __name__ == "__main__":
    import sys
    
    # Projekt-Root bestimmen
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    results_json_path = os.path.join(project_root, "data", "results", "results.json")
    plots_dir = os.path.join(project_root, "plots")
    
    print(f"{'='*80}")
    print("📊 PCA Dimension über Epochen - Plot Generator")
    print(f"{'='*80}")
    print(f"📁 Results JSON: {results_json_path}")
    print(f"📁 Output: {plots_dir}")
    print(f"{'='*80}\n")
    
    try:
        plot_pca_dimension_over_epochs(
            results_json_path=results_json_path,
            output_dir=plots_dir
        )
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

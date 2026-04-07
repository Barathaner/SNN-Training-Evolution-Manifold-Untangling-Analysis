"""
Skript zum Analysieren von Activity Logs und Erstellen von Plots.

Dieses Skript:
1. Lädt alle Activity Logs aus data/activity_logs
2. Führt Manifold-Analysen für jeden Log durch
3. Speichert Ergebnisse in JSON-Dateien
4. Erstellt alle Plots
"""

import os
import json
import re
import h5py
import numpy as np
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
import manifolduntanglinganalysis.analysis.intrinsic_dimension as id_analysis
from manifolduntanglinganalysis.metrics.mean_field_theoretic_manifold_analysis_wrapper import (
    analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds,
    analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded,
    plot_manifold_metrics_over_epochs,
    plot_manifold_metrics_over_layer,
    plot_manifold_metrics_over_epochs_all_layer_in_one_plot
)


def sort_key(filename):
    """Sortiere Activity Logs nach Epoche und Layer.
    
    Format: epoch_XXX_layername_spk_events.h5
    """
    match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', filename)
    if match:
        epoch = int(match.group(1))
        layer = match.group(2)
        # Sortiere zuerst nach Epoche, dann nach Layer
        return (epoch, layer)
    return (999, 'zzz')  # Unbekannte Dateien ans Ende


def load_input_data_metrics(results_dir):
    """Versuche input_data_metrics aus einer JSON-Datei zu laden, sonst verwende Standardwerte."""
    input_metrics_path = os.path.join(results_dir, "input_data_metrics.json")
    
    if os.path.exists(input_metrics_path):
        print(f"📂 Lade input_data_metrics aus {input_metrics_path}")
        with open(input_metrics_path, 'r') as f:
            return json.load(f)
    else:
        print("⚠️  input_data_metrics.json nicht gefunden, verwende Standardwerte")
        # Standardwerte (können angepasst werden)
        return {
            'capacity': 0.0069,
            'radius': 1.8510,
            'dimension': 187.0195,
            'correlation': 0.5949,
            'optimal_k': 2
        }


def analyze_activity_logs(
    activity_logs_path,
    results_dir,
    labels=None,
    max_samples_per_class=64,
    use_rate_coded_for_output=False,
    verbose=True
):
    """
    Analysiere alle Activity Logs und erstelle Ergebnisse.
    
    Args:
        activity_logs_path: Pfad zum Verzeichnis mit Activity Logs
        results_dir: Pfad zum Verzeichnis für Ergebnisse
        labels: Liste der zu analysierenden Labels (default: 0-19)
        max_samples_per_class: Maximale Anzahl Samples pro Klasse
        use_rate_coded_for_output: Wenn True, verwende rate_coded Analyse für Output-Layer
        verbose: Ausführliche Ausgabe
    """
    if labels is None:
        labels = list(range(0, 20))
    
    # Lade alle Activity Log Dateien
    activity_logs = [f for f in os.listdir(activity_logs_path) if f.endswith('.h5')]
    activity_logs = sorted(activity_logs, key=sort_key)
    
    if len(activity_logs) == 0:
        print(f"❌ Keine Activity Logs gefunden in {activity_logs_path}")
        return None
    
    print(f"📊 Gefunden: {len(activity_logs)} Activity Logs")
    
    results = {}  # Struktur: results[epoch][layer] = {'capacity': ..., 'radius': ..., 'dimension': ...}
    results_list = []  # Für plot_manifold_metrics_over_epochs
    
    for activity_log in activity_logs:
        # Parse Epoch und Layer aus dem Dateinamen
        match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', activity_log)
        if not match:
            print(f"⚠️  Warnung: Konnte Epoch und Layer nicht aus {activity_log} extrahieren")
            continue
        
        epoch = int(match.group(1))
        layer = match.group(2)
        
        # Lade Activity Log und erstelle Transform mit korrekter sensor_size
        activity_log_path = os.path.join(activity_logs_path, activity_log)
        
        try:
            with h5py.File(activity_log_path, 'r') as f:
                num_neurons = int(f.attrs['num_features'])
        except Exception as e:
            print(f"❌ Fehler beim Lesen von {activity_log}: {e}")
            continue
        
        # Erstelle Transform mit korrekter sensor_size
        activity_log_transform = datatransforms.get_activity_logpreprocessing(
            num_neurons=num_neurons,
            fixed_duration=80,
            n_time_bins=1
        )
        
        # Lade Activity Log mit Transform
        activity_log_dataloader = dataloader.load_activity_log(
            activity_log_path=activity_log_path,
            transform=activity_log_transform
        )
        
        print(f"\n{'='*80}")
        print(f"🔍 Analysiere: {activity_log}")
        print(f"   Epoch: {epoch}, Layer: {layer}, Neuronen: {num_neurons}")
        print(f"{'='*80}")
        
        # Führe Manifold-Analyse durch
        try:
            # Verwende rate_coded für Output-Layer, wenn gewünscht
            if use_rate_coded_for_output and "lif3" in layer:
                current_result = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded(
                    dataloader=activity_log_dataloader,
                    labels=labels,
                    max_samples_per_class=max_samples_per_class,
                    kappa=0.0,
                    n_t=200,
                    n_reps=1,
                    verbose=verbose
                )
            else:
                current_result = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
                    dataloader=activity_log_dataloader,
                    labels=labels,
                    max_samples_per_class=max_samples_per_class,
                    kappa=0.0,
                    n_t=200,
                    n_reps=1,
                    verbose=verbose
                )
        except Exception as e:
            print(f"❌ Fehler bei Manifold-Analyse für {activity_log}: {e}")
            continue
        
        # Berechne Intrinsic Dimension mit PCA
        try:
            plot_path = os.path.join(results_dir.replace("results", "plots"), 
                                    f"explained_variance_dimension_epoch_{epoch:03d}_{layer}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            pca_intdim, fig = id_analysis.explained_variance_dimension(
                activity_log_dataloader,
                perc=0.90,
                plot_path=plot_path
            )
            print(f"📐 PCA Intrinsic Dimension: {pca_intdim:.2f}")
        except Exception as e:
            print(f"⚠️  Fehler bei Intrinsic Dimension Berechnung: {e}")
            pca_intdim = 0.0
        
        # Zeige Ergebnisse
        print(f"📊 Capacity: {current_result['capacity']:.4f}")
        print(f"📊 Radius: {current_result['radius']:.4f}")
        print(f"📊 Dimension: {current_result['dimension']:.4f}")
        print(f"📊 Correlation: {current_result['correlation']:.4f}")
        
        # Strukturiere nach Epoch und Layer
        if epoch not in results:
            results[epoch] = {}
        
        # Konvertiere NumPy-Datentypen zu nativen Python-Typen
        results[epoch][layer] = {
            'capacity': float(current_result['capacity']),
            'radius': float(current_result['radius']),
            'dimension': float(current_result['dimension']),
            'correlation': float(current_result['correlation']),
            'pca_intdim': float(pca_intdim),
        }
        
        # Für plot_manifold_metrics_over_epochs die vollständigen Ergebnisse behalten
        results_list.append(current_result)
    
    return results


def save_results(results, results_dir):
    """Speichere Ergebnisse in JSON-Dateien."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Speichere results_all.json (wird für Plots benötigt)
    results_json_path = os.path.join(results_dir, "results_all.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Ergebnisse gespeichert: {results_json_path}")
    
    # Speichere auch results.json (für Kompatibilität)
    results_json_path_alt = os.path.join(results_dir, "results.json")
    with open(results_json_path_alt, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Ergebnisse gespeichert: {results_json_path_alt}")
    
    return results_json_path


def create_plots(results_json_path, input_data_metrics, plots_dir):
    """Erstelle alle Plots aus den Ergebnissen."""
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("📈 Erstelle Plots...")
    print(f"{'='*80}")
    
    try:
        plot_manifold_metrics_over_epochs(
            results_json_path=results_json_path,
            input_data_metrics=input_data_metrics,
            save_dir=plots_dir,
            figsize_per_subplot=(5, 4)
        )
        print("✅ Plot erstellt: manifold_metrics_over_epochs")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von manifold_metrics_over_epochs: {e}")
    
    try:
        plot_manifold_metrics_over_layer(
            results_json_path=results_json_path,
            input_data_metrics=input_data_metrics,
            save_dir=plots_dir,
            figsize_per_subplot=(5, 4)
        )
        print("✅ Plot erstellt: manifold_metrics_over_layer")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von manifold_metrics_over_layer: {e}")
    
    try:
        plot_manifold_metrics_over_epochs_all_layer_in_one_plot(
            results_json_path=results_json_path,
            input_data_metrics=input_data_metrics,
            save_dir=plots_dir,
            figsize_per_subplot=(5, 4)
        )
        print("✅ Plot erstellt: manifold_metrics_over_epochs_all_layer_in_one_plot")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von manifold_metrics_over_epochs_all_layer_in_one_plot: {e}")


if __name__ == "__main__":
    # Projekt-Root bestimmen
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Pfade definieren
    activity_logs_path = os.path.join(project_root, "data", "activity_logs_ffn")
    results_dir = os.path.join(project_root, "data", "results")
    plots_dir = os.path.join(project_root, "plots")
    
    print(f"{'='*80}")
    print("🚀 Activity Log Analyse Skript")
    print(f"{'='*80}")
    print(f"📁 Activity Logs: {activity_logs_path}")
    print(f"📁 Results: {results_dir}")
    print(f"📁 Plots: {plots_dir}")
    print(f"{'='*80}\n")
    
    # Prüfe ob Activity Logs Verzeichnis existiert
    if not os.path.exists(activity_logs_path):
        print(f"❌ Activity Logs Verzeichnis nicht gefunden: {activity_logs_path}")
        exit(1)
    
    # Lade input_data_metrics (oder verwende Standardwerte)
    input_data_metrics = load_input_data_metrics(results_dir)
    print(f"\n📊 Input Data Metrics:")
    print(f"   Capacity: {input_data_metrics['capacity']:.4f}")
    print(f"   Radius: {input_data_metrics['radius']:.4f}")
    print(f"   Dimension: {input_data_metrics['dimension']:.4f}")
    
    # Analysiere Activity Logs
    results = analyze_activity_logs(
        activity_logs_path=activity_logs_path,
        results_dir=results_dir,
        labels=list(range(0, 10)),  # Labels 0-19
        max_samples_per_class=64,
        use_rate_coded_for_output=False,  # Setze auf True, wenn Output-Layer rate_coded verwenden soll
        verbose=True
    )
    
    if results is None or len(results) == 0:
        print("\n❌ Keine Ergebnisse erzeugt. Beende Skript.")
        exit(1)
    
    # Speichere Ergebnisse
    results_json_path = save_results(results, results_dir)
    
    # Erstelle Plots
    create_plots(results_json_path, input_data_metrics, plots_dir)
    
    print(f"\n{'='*80}")
    print("✅ Analyse abgeschlossen!")
    print(f"{'='*80}")

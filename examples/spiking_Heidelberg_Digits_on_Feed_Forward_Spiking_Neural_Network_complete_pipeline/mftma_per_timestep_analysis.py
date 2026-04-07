"""
MFTMA pro Zeitschritt für alle Layer und alle Epochen.

Führt für jeden Activity-Log (epoch × layer) die Mean-Field Manifold-Analyse
für jeden Zeitschritt einzeln durch und speichert die Ergebnisse in einer JSON:
  - manifold capacity (lower bound)
  - manifold radius
  - manifold dimension (upper bound)
  - center correlation

JSON-Struktur (für Plot-Skripte):
  {
    "0": { "lif0": { "timesteps": [0,1,...], "capacity": [...], "radius": [...], "dimension": [...], "correlation": [...] }, ... },
    "1": { ... },
    ...
  }
  Epoch-Key als String, pro Layer: timesteps, capacity, radius, dimension, correlation (Listen gleicher Länge).

Verwendung:
  python mftma_per_timestep_analysis.py [--activity-logs DIR] [--out FILE] [--time-bins T] [--max-samples N]
"""

import os
import sys
import re
import json
import argparse
import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
from manifolduntanglinganalysis.metrics.mean_field_theoretic_manifold_analysis_wrapper import (
    analyze_manifold_metrics_per_timestep,
)


def _to_serializable(obj):
    """Convert numpy types to native Python for JSON (NaN → None)."""
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def sort_key(filename):
    """Sort activity logs by epoch then layer."""
    match = re.match(r"epoch_(\d+)_(\w+)_spk_events\.h5", filename)
    if match:
        return (int(match.group(1)), match.group(2))
    return (999, "zzz")


def main():
    parser = argparse.ArgumentParser(
        description="MFTMA pro Zeitschritt für alle Layer und Epochen → JSON"
    )
    parser.add_argument(
        "--activity-logs",
        type=str,
        default=None,
        help="Verzeichnis mit Activity-Log H5-Dateien (Standard: project/data/activity_logs_ffn)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Ausgabe-JSON (Standard: project/data/results/mftma_per_timestep.json)",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=40,
        help="Anzahl Zeitschritte/Bins pro Sample (Standard: 80)",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=50,
        help="Max. Samples pro Klasse für MFTMA (Standard: 50)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="0-9",
        help="Labels als Bereich, z.B. 0-9 oder 0-19 (Standard: 0-9)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="MFTMA margin kappa (Standard: 0.0)",
    )
    parser.add_argument(
        "--n-t",
        type=int,
        default=200,
        help="MFTMA n_t (Standard: 200)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Ausführliche Ausgabe",
    )
    args = parser.parse_args()

    activity_logs_path = args.activity_logs or os.path.join(
        PROJECT_ROOT, "data", "activity_logs_ffn"
    )
    out_path = args.out or os.path.join(
        PROJECT_ROOT, "data", "results", "mftma_per_timestep.json"
    )

    # Parse labels (e.g. "0-9" -> [0,1,...,9])
    if "-" in args.labels:
        a, b = args.labels.split("-")
        labels = list(range(int(a), int(b) + 1))
    else:
        labels = [int(x) for x in args.labels.split(",")]
    if len(labels) < 2:
        print("Mindestens 2 Labels nötig.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(activity_logs_path):
        print(f"Activity-Logs-Verzeichnis nicht gefunden: {activity_logs_path}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(
        [f for f in os.listdir(activity_logs_path) if f.endswith(".h5")],
        key=sort_key,
    )
    if not log_files:
        print(f"Keine .h5 Activity Logs in {activity_logs_path}", file=sys.stderr)
        sys.exit(1)

    # Group by layer, keep list of paths per layer (one per epoch)
    layer_paths = {}
    layer_neurons = {}
    for f in log_files:
        m = re.match(r"epoch_(\d+)_(\w+)_spk_events\.h5", f)
        if not m:
            continue
        epoch, layer = int(m.group(1)), m.group(2)
        path = os.path.join(activity_logs_path, f)
        if layer not in layer_paths:
            layer_paths[layer] = []
            try:
                with h5py.File(path, "r") as h5:
                    layer_neurons[layer] = int(h5.attrs["num_features"])
            except Exception as e:
                print(f"Warnung: num_features für {layer} nicht lesbar: {e}")
                layer_neurons[layer] = 350
        layer_paths[layer].append((epoch, path))

    for layer in layer_paths:
        layer_paths[layer].sort(key=lambda x: x[0])

    print(f"Gefunden: {len(log_files)} Logs, Layer: {list(layer_paths.keys())}")
    print(f"Zeitschritte pro Sample: {args.time_bins}, Labels: {labels}")
    print()

    # Results: nested dict for JSON
    # results[epoch][layer] = { "timesteps": [...], "capacity": [...], "radius": [...], "dimension": [...], "correlation": [...] }
    results = {}

    for layer, epoch_paths in layer_paths.items():
        num_neurons = layer_neurons.get(layer, 350)
        transform = datatransforms.get_activity_logpreprocessing(
            num_neurons=num_neurons,
            fixed_duration=float(args.time_bins),
            n_time_bins=args.time_bins,
        )

        for epoch, activity_log_path in epoch_paths:
            if args.verbose:
                print(f"[Epoch {epoch}] Layer {layer} …")

            try:
                dl = dataloader.load_activity_log(
                    activity_log_path=activity_log_path,
                    transform=transform,
                    batch_size=64,
                    drop_last=False,
                    num_workers=0,
                )
            except Exception as e:
                print(f"  Fehler beim Laden von {activity_log_path}: {e}")
                continue

            try:
                out = analyze_manifold_metrics_per_timestep(
                    dataloader=dl,
                    labels=labels,
                    max_samples_per_class=args.max_samples_per_class,
                    kappa=args.kappa,
                    n_t=args.n_t,
                    n_reps=1,
                    verbose=args.verbose,
                )
            except Exception as e:
                print(f"  Fehler bei MFTMA: {e}")
                import traceback
                traceback.print_exc()
                continue

            if str(epoch) not in results:
                results[str(epoch)] = {}
            results[str(epoch)][layer] = {
                "timesteps": out["timesteps"],
                "capacity": _to_serializable(out["capacity"]),
                "radius": _to_serializable(out["radius"]),
                "dimension": _to_serializable(out["dimension"]),
                "correlation": _to_serializable(out["correlation"]),
            }
            if args.verbose:
                print(f"  → {out['n_timesteps']} Timesteps gespeichert.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nErgebnisse geschrieben: {out_path}")


if __name__ == "__main__":
    main()

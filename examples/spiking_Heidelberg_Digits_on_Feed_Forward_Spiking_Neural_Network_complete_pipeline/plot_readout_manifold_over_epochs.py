"""
Plottet den Anstieg der Manifold-Capacity und der Center-Correlation des Readout-Layers (lif3)
über die Epochen. Erzeugt zwei separate Bilddateien.
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Name des Readout-Layers (letzter versteckter LIF-Layer vor der Klassifikation)
READOUT_LAYER = "lif3"


def load_readout_metrics(results_json_path: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Lädt Capacity und Center-Correlation des Readout-Layers pro Epoch aus results.json.

    Erwartet Struktur: {"epoch_str": {"layer_name": {"capacity": float, "correlation": float, ...}}, ...}

    Returns:
        epochs: Liste der Epochen (aufsteigend)
        capacities: Capacity pro Epoch
        correlations: Center-Correlation pro Epoch
    """
    with open(results_json_path, "r") as f:
        data = json.load(f)

    epochs = []
    capacities = []
    correlations = []

    for epoch_str in sorted(data.keys(), key=int):
        layers = data[epoch_str]
        if READOUT_LAYER not in layers:
            continue
        metrics = layers[READOUT_LAYER]
        epochs.append(int(epoch_str))
        capacities.append(float(metrics["capacity"]))
        correlations.append(float(metrics.get("correlation", np.nan)))

    return epochs, capacities, correlations


def plot_capacity_over_epochs(
    epochs: List[int],
    capacities: List[float],
    save_path: Path,
    figsize: Tuple[float, float] = (7, 4),
) -> None:
    """Plottet Manifold-Capacity des Readout-Layers über Epochen und speichert als Datei."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, capacities, marker="o", linestyle="-", linewidth=2, markersize=6)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Manifold Capacity (α_M)", fontsize=12)
    ax.set_title("Manifold Capacity des Readout-Layers (lif3) über Epochen", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(epochs) - 0.2, right=max(epochs) + 0.2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_center_correlation_over_epochs(
    epochs: List[int],
    correlations: List[float],
    save_path: Path,
    figsize: Tuple[float, float] = (7, 4),
) -> None:
    """Plottet Center-Correlation des Readout-Layers über Epochen und speichert als Datei."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, correlations, marker="s", linestyle="-", linewidth=2, markersize=6, color="C1")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Center Correlation", fontsize=12)
    ax.set_title("Center Correlation des Readout-Layers (lif3) über Epochen", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(epochs) - 0.2, right=max(epochs) + 0.2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plottet Manifold-Capacity und Center-Correlation des Readout-Layers über Epochen."
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--results",
        type=Path,
        default=project_root / "data" / "results" / "results.json",
        help="Pfad zu results.json (Default: data/results/results.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "plots",
        help="Verzeichnis für die PNG-Dateien (Default: plots/)",
    )
    parser.add_argument(
        "--capacity-name",
        type=str,
        default="readout_manifold_capacity_over_epochs.png",
        help="Dateiname für den Capacity-Plot (Default: readout_manifold_capacity_over_epochs.png)",
    )
    parser.add_argument(
        "--correlation-name",
        type=str,
        default="readout_center_correlation_over_epochs.png",
        help="Dateiname für den Center-Correlation-Plot (Default: readout_center_correlation_over_epochs.png)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results-Datei nicht gefunden: {args.results}")

    epochs, capacities, correlations = load_readout_metrics(args.results)
    if not epochs:
        raise ValueError(
            f"Keine Daten für Readout-Layer '{READOUT_LAYER}' in {args.results} gefunden."
        )

    out_dir = Path(args.out_dir)
    capacity_path = out_dir / args.capacity_name
    correlation_path = out_dir / args.correlation_name

    plot_capacity_over_epochs(epochs, capacities, capacity_path)
    print(f"Capacity-Plot gespeichert: {capacity_path}")

    plot_center_correlation_over_epochs(epochs, correlations, correlation_path)
    print(f"Center-Correlation-Plot gespeichert: {correlation_path}")


if __name__ == "__main__":
    main()

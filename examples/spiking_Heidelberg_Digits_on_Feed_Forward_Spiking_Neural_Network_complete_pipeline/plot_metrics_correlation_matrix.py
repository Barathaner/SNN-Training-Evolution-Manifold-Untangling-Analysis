"""
Erstellt eine Korrelationsmatrix zwischen Manifold-Metriken (Readout-Layer) und
allen Performance-Metriken über die Epochen und speichert sie als Plot.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


READOUT_LAYER = "lif3"

# Anzeigenamen für die Spalten (kürzer für die Achsen)
DISPLAY_NAMES = {
    "capacity": "Manifold Capacity",
    "radius": "Manifold Radius",
    "dimension": "Manifold Dimension",
    "correlation": "Center Correlation",
    "train_loss": "Train Loss",
    "train_accuracy": "Train Accuracy",
    "val_loss": "Val Loss",
    "val_accuracy": "Val Accuracy",
    "val_precision": "Val Precision",
    "val_recall": "Val Recall",
    "val_f1": "Val F1",
    "val_auc_roc": "Val AUC-ROC",
    "silhouette_score": "Silhouette-Score",
}


def load_manifold_readout_per_epoch(results_path: Path) -> Dict[str, List[float]]:
    """
    Lädt aus results.json für jede Epoch die Metriken des Readout-Layers (lif3).
    Returns: {"capacity": [..], "radius": [..], "dimension": [..], "correlation": [..]}
    mit Listen sortiert nach Epoch.
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    out: Dict[str, List[float]] = {
        "capacity": [],
        "radius": [],
        "dimension": [],
        "correlation": [],
    }
    for epoch_str in sorted(data.keys(), key=int):
        layers = data[epoch_str]
        if READOUT_LAYER not in layers:
            continue
        m = layers[READOUT_LAYER]
        out["capacity"].append(float(m["capacity"]))
        out["radius"].append(float(m["radius"]))
        out["dimension"].append(float(m["dimension"]))
        out["correlation"].append(float(m.get("correlation", np.nan)))

    return out


def load_performance_metrics(perf_path: Path) -> Dict[str, List[float]]:
    """
    Lädt performance_metrics.json.
    Returns: {"train_loss": [..], "train_accuracy": [..], ...}
    """
    with open(perf_path, "r") as f:
        data = json.load(f)

    # Alle Keys außer "epochs" als Metriken
    return {
        k: [float(x) for x in v]
        for k, v in data.items()
        if k != "epochs" and isinstance(v, list)
    }


def load_silhouette_scores(
    silhouette_json_path: Optional[Path] = None,
    activity_logs_dir: Optional[Path] = None,
    layer: str = READOUT_LAYER,
    n_epochs: int = 10,
    verbose: bool = True,
) -> Optional[List[float]]:
    """
    Silhouette-Scores pro Epoch (Reihenfolge 1, 2, ..., n_epochs).
    Entweder aus JSON laden (Keys: "epochs", "silhouette_score") oder
    per compute_silhouette_per_epoch aus Activity-Logs berechnen.
    """
    if silhouette_json_path is not None and silhouette_json_path.exists():
        with open(silhouette_json_path, "r") as f:
            data = json.load(f)
        epochs_in = data.get("epochs", [])
        scores_in = data.get("silhouette_score", data.get("scores", []))
        by_epoch = dict(zip(epochs_in, scores_in))
        out = [by_epoch.get(e, np.nan) for e in range(1, n_epochs + 1)]
        return out

    if activity_logs_dir is not None and activity_logs_dir.is_dir():
        try:
            from plot_silhouette_score_over_epochs import compute_silhouette_per_epoch
        except ImportError:
            if verbose:
                print("Hinweis: Silhouette nicht berechnet (plot_silhouette_score_over_epochs nicht importierbar).")
            return None
        epochs_list, scores_list = compute_silhouette_per_epoch(
            activity_logs_dir, layer=layer, verbose=verbose
        )
        by_epoch = dict(zip(epochs_list, scores_list))
        out = [by_epoch.get(e, np.nan) for e in range(1, n_epochs + 1)]
        return out

    return None


def build_combined_table(
    manifold: Dict[str, List[float]],
    performance: Dict[str, List[float]],
    silhouette_scores: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Baut eine Matrix (n_epochs, n_metrics) und die Liste der Metrik-Namen.
    Epochs müssen übereinstimmen (gleiche Länge).
    """
    n_epochs_man = len(manifold["capacity"])
    n_epochs_perf = len(performance.get("train_loss", performance.get("val_accuracy", [])))
    n_epochs = min(n_epochs_man, n_epochs_perf)
    if n_epochs == 0:
        raise ValueError("Keine gemeinsamen Epochen zwischen Manifold- und Performance-Daten.")

    # Manifold-Metriken (Readout)
    columns: List[str] = ["capacity", "radius", "dimension", "correlation"]
    rows: List[List[float]] = []
    for i in range(n_epochs):
        row = [
            manifold["capacity"][i],
            manifold["radius"][i],
            manifold["dimension"][i],
            manifold["correlation"][i],
        ]
        if silhouette_scores is not None and len(silhouette_scores) >= i + 1:
            row.append(silhouette_scores[i])
        rows.append(row)

    if silhouette_scores is not None and len(silhouette_scores) >= n_epochs:
        columns.append("silhouette_score")

    # Performance-Metriken (alle die in performance sind)
    perf_keys = [k for k in performance if k != "epochs" and isinstance(performance[k], list)]
    for k in perf_keys:
        columns.append(k)
        for i in range(n_epochs):
            if i < len(rows):
                val = performance[k][i] if i < len(performance[k]) else np.nan
                rows[i].append(float(val))

    matrix = np.array(rows, dtype=np.float64)
    return matrix, columns


def plot_correlation_matrix(
    matrix: np.ndarray,
    column_names: List[str],
    save_path: Path,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Berechnet die Korrelationsmatrix und plottet sie als Heatmap."""
    # NaN-Zeilen entfernen oder mit 0 füllen für Korrelation
    mask = ~np.any(np.isnan(matrix), axis=1)
    if not np.all(mask):
        matrix = matrix[mask]

    if matrix.shape[0] < 2:
        raise ValueError("Mindestens 2 Epochen nötig für Korrelationsmatrix.")

    corr = np.corrcoef(matrix.T)
    # Bei konstanten Spalten kann NaN entstehen
    np.nan_to_num(corr, copy=False, nan=0.0)

    labels = [DISPLAY_NAMES.get(c, c) for c in column_names]

    if figsize is None:
        n = len(labels)
        figsize = (max(8, n * 0.7), max(6, n * 0.55))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "—"
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Korrelation", shrink=0.8)
    ax.set_title("Korrelationsmatrix: Manifold-Metriken (Readout) & Performance-Metriken", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plottet Korrelationsmatrix zwischen Manifold- und Performance-Metriken."
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--results",
        type=Path,
        default=project_root / "data" / "results" / "results.json",
        help="Pfad zu results.json (Manifold-Metriken)",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=project_root / "data" / "results" / "performance_metrics.json",
        help="Pfad zu performance_metrics.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "plots" / "metrics_correlation_matrix.png",
        help="Ausgabepfad für den Plot",
    )
    default_logs = project_root / "data" / "activity_logs"
    if not default_logs.is_dir():
        default_logs = project_root / "data" / "activity_logs_feed_forward"
    parser.add_argument(
        "--silhouette-json",
        type=Path,
        default=project_root / "data" / "results" / "silhouette_scores.json",
        help="JSON mit Silhouette-Scores (epochs, silhouette_score). Wenn vorhanden, wird geladen.",
    )
    parser.add_argument(
        "--activity-logs",
        type=Path,
        default=None,
        help="Falls gesetzt und --silhouette-json fehlt: Silhouette aus Activity-Logs berechnen.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=READOUT_LAYER,
        help="Layer für Silhouette-Berechnung (Default: lif3)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results-Datei nicht gefunden: {args.results}")
    if not args.performance.exists():
        raise FileNotFoundError(f"Performance-Datei nicht gefunden: {args.performance}")

    manifold = load_manifold_readout_per_epoch(args.results)
    performance = load_performance_metrics(args.performance)
    n_epochs = min(len(manifold["capacity"]), len(performance.get("train_loss", [])))

    # Silhouette: zuerst JSON, sonst aus Activity-Logs berechnen
    activity_logs = args.activity_logs if args.activity_logs is not None else default_logs
    silhouette_scores = load_silhouette_scores(
        silhouette_json_path=args.silhouette_json,
        activity_logs_dir=activity_logs if activity_logs.is_dir() else None,
        layer=args.layer,
        n_epochs=n_epochs,
        verbose=True,
    )
    if silhouette_scores is None:
        print("Hinweis: Silhouette-Score nicht in Matrix (keine Daten). Optional: --activity-logs setzen oder silhouette_scores.json anlegen.")

    matrix, columns = build_combined_table(manifold, performance, silhouette_scores)
    plot_correlation_matrix(matrix, columns, args.out)
    print(f"Korrelationsmatrix gespeichert: {args.out}")


if __name__ == "__main__":
    main()

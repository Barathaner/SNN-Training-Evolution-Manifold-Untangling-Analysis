"""
Berechnet und plottet den Silhouette-Score pro Epoch für einen Layer (Standard: Readout lif3).
Nutzt Activity-Log H5-Dateien: pro Epoch werden die Aktivitäten als Feature-Vektoren (pro Sample)
gesammelt und der Silhouette-Score bezüglich der echten Klassen-Labels berechnet.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader


def sort_key(filename: str) -> Tuple[int, str]:
    """Sortiere Activity-Logs nach Epoch und Layer (epoch_XXX_layername_spk_events.h5)."""
    match = re.match(r"epoch_(\d+)_(\w+)_spk_events\.h5", filename)
    if match:
        return (int(match.group(1)), match.group(2))
    return (999, "zzz")


def collect_X_and_labels_for_epoch(
    activity_log_path: str,
    num_neurons: int,
    n_time_bins: int = 1,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt einen Activity-Log, transformiert zu Frames und liefert eine Feature-Matrix X
    (n_samples, n_features) und Labels (n_samples,). Pro Sample wird die zeitliche
    Aktivität zu einem Vektor flach gemacht (alle Time-Bins aneinandergereiht).

    Returns:
        X: (n_samples, n_neurons * n_time_bins), float
        labels: (n_samples,), int
    """
    transform = datatransforms.get_activity_logpreprocessing(
        num_neurons=num_neurons,
        fixed_duration=80,
        n_time_bins=n_time_bins,
    )
    dl = dataloader.load_activity_log(
        activity_log_path=activity_log_path,
        transform=transform,
        drop_last=False,
    )
    X_list: List[np.ndarray] = []
    label_list: List[int] = []
    for batch_data, batch_labels in dl:
        batch_data_np = (
            batch_data.numpy() if hasattr(batch_data, "numpy") else np.array(batch_data)
        )
        batch_labels_np = (
            batch_labels.numpy()
            if hasattr(batch_labels, "numpy")
            else np.array(batch_labels)
        )
        for i in range(batch_data_np.shape[0]):
            frames = batch_data_np[i]
            if frames.ndim == 4:
                # (T, C, H, W) -> (T, N)
                vec = frames.reshape(frames.shape[0], -1)
            elif frames.ndim == 3:
                vec = frames[:, 0, :] if frames.shape[1] == 1 else frames
            else:
                vec = np.asarray(frames)
            # Ein Vektor pro Sample: alle Zeitschritte flach
            x_flat = vec.ravel().astype(np.float64)
            X_list.append(x_flat)
            label_list.append(int(batch_labels_np[i]))
            if max_samples is not None and len(X_list) >= max_samples:
                break
        if max_samples is not None and len(X_list) >= max_samples:
            break

    X = np.stack(X_list, axis=0)
    labels = np.array(label_list, dtype=np.int32)
    return X, labels


def compute_silhouette_per_epoch(
    activity_logs_dir: Path,
    layer: str = "lif3",
    n_time_bins: int = 1,
    max_samples_per_epoch: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[int], List[float]]:
    """
    Findet alle Activity-Logs für den angegebenen Layer, gruppiert nach Epoch,
    berechnet pro Epoch den Silhouette-Score und gibt Epochen und Scores zurück.
    """
    if not activity_logs_dir.is_dir():
        raise FileNotFoundError(f"Activity-Logs-Verzeichnis nicht gefunden: {activity_logs_dir}")

    h5_files = sorted(
        [f for f in activity_logs_dir.iterdir() if f.suffix == ".h5"],
        key=lambda p: sort_key(p.name),
    )
    # Nur Dateien für den gewünschten Layer
    epoch_paths: List[Tuple[int, Path]] = []
    layer_neurons: Optional[int] = None
    for p in h5_files:
        m = re.match(r"epoch_(\d+)_(\w+)_spk_events\.h5", p.name)
        if not m or m.group(2) != layer:
            continue
        epoch = int(m.group(1))
        if layer_neurons is None:
            try:
                with h5py.File(p, "r") as f:
                    layer_neurons = int(f.attrs["num_features"])
            except Exception as e:
                if verbose:
                    print(f"Warnung: num_features für {p.name} nicht lesbar: {e}")
                layer_neurons = 350
        epoch_paths.append((epoch, p))

    if not epoch_paths:
        raise ValueError(
            f"Keine Activity-Logs für Layer '{layer}' in {activity_logs_dir} gefunden."
        )
    if layer_neurons is None:
        layer_neurons = 350

    epoch_numbers = sorted({e for e, _ in epoch_paths})
    if verbose:
        print(f"Verzeichnis: {activity_logs_dir}")
        print(f"Gefunden: {len(epoch_paths)} Activity-Logs für {layer} (Epochs {min(epoch_numbers)}–{max(epoch_numbers)}).")

    epochs: List[int] = []
    scores: List[float] = []
    for epoch, path in sorted(epoch_paths, key=lambda x: x[0]):
        score_val: Optional[float] = None
        try:
            X, labels = collect_X_and_labels_for_epoch(
                str(path),
                num_neurons=layer_neurons,
                n_time_bins=n_time_bins,
                max_samples=max_samples_per_epoch,
            )
        except Exception as e:
            if verbose:
                print(f"Epoch {epoch}: Fehler beim Laden – {e}")
            epochs.append(epoch)
            scores.append(np.nan)
            continue

        n_classes = len(np.unique(labels))
        if n_classes < 2:
            if verbose:
                print(f"Epoch {epoch}: Nur eine Klasse vorhanden (n={len(labels)}), überspringe Silhouette.")
            epochs.append(epoch)
            scores.append(np.nan)
            continue

        try:
            sample_size = min(5000, len(X)) if len(X) > 5000 else None
            score_val = float(silhouette_score(X, labels, metric="euclidean", sample_size=sample_size))
        except Exception as e:
            if verbose:
                print(f"Epoch {epoch}: Silhouette-Berechnung fehlgeschlagen – {e}")
            epochs.append(epoch)
            scores.append(np.nan)
            continue

        epochs.append(epoch)
        scores.append(score_val)
        if verbose:
            print(f"Epoch {epoch}: Silhouette-Score = {score_val:.4f} (n={X.shape[0]}, Klassen={n_classes})")

    return epochs, scores


def plot_silhouette_over_epochs(
    epochs: List[int],
    scores: List[float],
    save_path: Path,
    figsize: Tuple[float, float] = (7, 4),
) -> None:
    """Plottet Silhouette-Score über Epochen und speichert als Datei. NaN-Scores werden ausgelassen (Lücke in der Linie)."""
    fig, ax = plt.subplots(figsize=figsize)
    scores_arr = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores_arr)
    ax.plot(np.array(epochs)[valid], scores_arr[valid], marker="o", linestyle="-", linewidth=2, markersize=6)
    # Alle Epochs auf X-Achse anzeigen (auch wenn manche fehlen), damit 1–10 sichtbar sind
    if epochs:
        ax.set_xlim(left=min(epochs) - 0.2, right=max(epochs) + 0.2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Silhouette-Score", fontsize=12)
    ax.set_title("Silhouette-Score über Epochen (Klassentrennung)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Berechnet und plottet den Silhouette-Score pro Epoch."
    )
    project_root = Path(__file__).resolve().parents[2]
    # Bevorzuge activity_logs_feed_forward (hat meist alle Epochen); sonst activity_logs
    default_logs = project_root / "data" / "activity_logs_feed_forward"
    if not default_logs.is_dir():
        default_logs = project_root / "data" / "activity_logs"

    parser.add_argument(
        "--activity-logs",
        type=Path,
        default=default_logs,
        help="Verzeichnis mit Activity-Log H5-Dateien (Standard: data/activity_logs_feed_forward)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="lif3",
        choices=["lif0", "lif1", "lif2", "lif3"],
        help="Layer für die Aktivitäten (Default: lif3 = Readout)",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=80,
        help="Anzahl Time-Bins für die Transform (Default: 10)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max. Samples pro Epoch (Default: alle)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "plots" / "silhouette_score_over_epochs.png",
        help="Ausgabepfad für den Plot",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Weniger Ausgabe",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Silhouette-Scores als JSON speichern (epochs, silhouette_score) für Korrelationsmatrix-Skript.",
    )
    args = parser.parse_args()

    epochs, scores = compute_silhouette_per_epoch(
        args.activity_logs,
        layer=args.layer,
        n_time_bins=args.time_bins,
        max_samples_per_epoch=args.max_samples,
        verbose=not args.quiet,
    )

    if not epochs:
        raise SystemExit("Keine gültigen Silhouette-Scores berechnet.")

    n_valid = sum(1 for s in scores if np.isfinite(s))
    if n_valid < len(epochs) and not args.quiet:
        print(f"Hinweis: Nur {n_valid}/{len(epochs)} Epochen mit gültigem Score (fehlende: Epochs ohne Daten oder nur eine Klasse).")

    plot_silhouette_over_epochs(epochs, scores, args.out)
    print(f"Plot gespeichert: {args.out}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"epochs": epochs, "silhouette_score": scores}, f, indent=2)
        print(f"JSON gespeichert: {args.save_json}")


if __name__ == "__main__":
    main()

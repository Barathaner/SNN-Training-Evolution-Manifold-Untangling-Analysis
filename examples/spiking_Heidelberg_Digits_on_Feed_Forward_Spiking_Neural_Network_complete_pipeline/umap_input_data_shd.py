"""
2D- oder 3D-UMAP-Visualisierung der SHD-Trainingsdaten (Input, vorverarbeitet).

- Preprocessing wie in der Pipeline: wählbare Zeitbins (Standard 80), 350 Neuronen (Downsample von 700)
- Nur Labels 0–9 (10 Klassen)
- 100 Samples pro Klasse (1000 Samples gesamt)
- Vier Plots (2×2): (1) Label, (2) Speaker-ID, (3) Speaker-Geschlecht, (4) Trajektorien (Zeitbins eingefärbt).

Verwendung:
  python umap_input_data_shd.py [--time-bins N] [--dim 2|3] [--out plots/umap_input_shd.png]
  Optional: --drop-empty-time-bins entfernt Zeitbins ohne Spikes (nur Nullen) vor dem UMAP-Fit.
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader

try:
    import umap
except ImportError:
    print("UMAP nicht installiert. Bitte installieren: pip install umap-learn", file=sys.stderr)
    sys.exit(1)


def get_colors_for_labels(n_labels=10):
    """Hochkontrast-Farben für die Klassen 0–9 (stark gesättigt, gut unterscheidbar)."""
    import colorsys
    colors = []
    for i in range(n_labels):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio für maximale Trennung
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append(mcolors.to_rgba(rgb))
    return colors


def get_speaker_high_contrast_cmap(n_colors):
    """Maximal unterscheidbare Farben für Speaker (keine ähnlichen Blautöne)."""
    import colorsys
    colors = []
    for i in range(max(n_colors, 1)):
        hue = (i * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        colors.append(mcolors.rgb2hex(rgb))
    return mcolors.ListedColormap(colors[:n_colors])


def load_speaker_and_gender_for_indices(data_path: str, original_indices: np.ndarray):
    """
    Lädt Speaker-IDs und Geschlecht für die gegebenen Dataset-Indizes aus der SHD-H5-Datei.
    Returns:
        speaker_ids: (N,) int
        genders: (N,) str, 'male' oder 'female'
    """
    base = Path(data_path)
    for h5_path in [base / "SHD" / "shd_train.h5", base / "shd_train.h5"]:
        if not h5_path.is_file():
            continue
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                all_speakers = f["extra"]["speaker"][:]
                gender_raw = f["extra"]["meta_info"]["gender"][:]
            speaker_genders = [g.decode("utf-8") if hasattr(g, "decode") else str(g) for g in gender_raw]
            speaker_ids = all_speakers[original_indices]
            genders = np.array([speaker_genders[int(sid)] for sid in speaker_ids])
            return speaker_ids, genders
        except Exception:
            continue
    return None, None


def main():
    parser = argparse.ArgumentParser(description="UMAP 2D/3D der SHD-Input-Daten (preprocessed, wählbare Zeitbins, 100 pro Klasse)")
    parser.add_argument("--data-path", type=str, default=None, help="Pfad zu SHD-Daten (Standard: project/data/input)")
    parser.add_argument("--time-bins", type=int, default=1, help="Anzahl Zeitbins im Preprocessing (Standard: 80)")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Samples pro Klasse (Standard: 100)")
    parser.add_argument("--n-neighbors", type=int, default=10, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.2, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3], help="UMAP-Dimension: 2D oder 3D (Standard: 2)")
    parser.add_argument("--max-points", type=int, default=10000, help="Max. Punkte im Plot (nach UMAP Downsampling; Standard: 10000)")
    parser.add_argument("--drop-empty-time-bins", action="store_true", help="Leere Zeitbins (nur Nullen in allen Neuronen) vor UMAP-Fit entfernen")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample-Index (0 bis N-1) für Einzel-Trajektorien-Plot (Standard: 0)")
    parser.add_argument("--out", type=str, default=None, help="Ausgabepfad (Standard: plots/umap_input_shd.png)")
    args = parser.parse_args()

    n_time_bins = args.time_bins

    data_path = args.data_path or os.path.join(PROJECT_ROOT, "data", "input")
    # Eindeutigen Suffix aus Parametern für Dateinamen
    param_suffix = (
        f"tb{args.time_bins}_spc{args.samples_per_class}_nn{args.n_neighbors}"
        f"_md{args.min_dist}_s{args.seed}_d{args.dim}_mp{args.max_points}"
    )
    if args.drop_empty_time_bins:
        param_suffix += "_drop"
    out_dir = os.path.dirname(args.out) if args.out else os.path.join(PROJECT_ROOT, "plots")
    if not out_dir:
        out_dir = "."
    base_name = f"umap_input_shd_{param_suffix}"
    out_path = args.out if args.out else os.path.join(out_dir, f"{base_name}.png")
    csv_path = os.path.join(out_dir, f"umap_embeddings_{param_suffix}.csv")
    use_3d = args.dim == 3

    # Gefilterte Indizes (gleiche Reihenfolge wie im Dataloader)
    import tonic
    dataset_full = tonic.datasets.SHD(save_to=data_path, train=True, transform=None)
    filtered_indices = np.array([i for i in range(len(dataset_full)) if dataset_full[i][1] in range(10)])
    del dataset_full

    # Preprocessing: wählbare Zeitbins, 350 Neuronen
    transform = datatransforms.get_preprocessing(
        n_time_bins=n_time_bins,
        target_neurons=350,
        original_neurons=700,
        fixed_duration=958007.0,
    )
    # Wichtig: Cache im Dataloader nutzt transform.n_time_bins; Compose hat das nicht → immer "default" = alter 80-Bins-Cache.
    # Setzen, damit bei --time-bins 1 (oder anders) ein eigener Cache-Pfad verwendet wird.
    transform.n_time_bins = n_time_bins

    train_loader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10),
        data_path=data_path,
        transform=transform,
        train=True,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )

    # Sammle alle Samples, Labels und Original-Indizes (für Speaker/Gender)
    all_data = []
    all_labels = []
    all_original_indices = []
    pos = 0
    for events, labels in train_loader:
        if events.ndim == 4:
            events = events.squeeze(2)
        events_np = events.numpy() if isinstance(events, torch.Tensor) else events
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        batch_size = events_np.shape[0]
        for i in range(batch_size):
            all_data.append(events_np[i])
            all_labels.append(int(labels_np[i]))
            all_original_indices.append(filtered_indices[pos])
            pos += 1
        if pos >= len(filtered_indices):
            break

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_original_indices = np.array(all_original_indices)
    print(f"Gesamt geladene Samples: {len(all_labels)} (Labels 0–9)")

    # 100 Samples pro Klasse (zufällig auswählen)
    np.random.seed(args.seed)
    indices_per_class = {}
    for c in range(10):
        indices_per_class[c] = np.where(all_labels == c)[0]
    selected = []
    for c in range(10):
        idx = indices_per_class[c]
        if len(idx) >= args.samples_per_class:
            chosen = np.random.choice(idx, size=args.samples_per_class, replace=False)
        else:
            chosen = idx
        selected.extend(chosen)
    selected = np.array(selected)
    X_raw = all_data[selected]
    labels = all_labels[selected]
    original_indices = all_original_indices[selected]

    # Stichprobenanzahl prüfen und ausgeben
    n_total = len(labels)
    counts_per_class = [np.sum(labels == c) for c in range(10)]
    print(f"Samples pro Klasse nach Auswahl: {counts_per_class} (Summe = {n_total})")
    if n_total != 10 * args.samples_per_class and all(c >= args.samples_per_class for c in counts_per_class):
        print(f"  (Erwartet: 10 × {args.samples_per_class} = {10 * args.samples_per_class})")
    elif any(c < args.samples_per_class for c in counts_per_class):
        print(f"  Hinweis: Einige Klassen haben < {args.samples_per_class} Samples im Trainingsset.")

    # Speaker und Geschlecht für die ausgewählten Samples
    speaker_ids, genders = load_speaker_and_gender_for_indices(data_path, original_indices)
    if speaker_ids is None:
        print("Hinweis: Speaker/Gender aus H5 nicht geladen, nur Label-Plot wird erstellt.")
    else:
        speaker_ids = np.asarray(speaker_ids)
        genders = np.asarray(genders)

    # Einheitlich Trajektorien: ein Punkt pro Zeitbin pro Sample – alle 4 Plots nutzen dasselbe Embedding.
    X_flat = X_raw.reshape(-1, X_raw.shape[2])  # (N*n_time_bins, 350)
    n_pts_full = X_flat.shape[0]
    # Optional: leere Zeitbins (keine Spikes) vor UMAP-Fit entfernen
    if args.drop_empty_time_bins:
        empty_bins = (X_raw == 0).all(axis=2)   # (N, n_time_bins), True wo Zeitbin nur Nullen
        keep_flat = ~empty_bins.ravel()
        n_dropped = int(np.sum(~keep_flat))
        X_traj = X_flat[keep_flat]
        flat_indices_kept = np.where(keep_flat)[0]
        sample_index_per_point = flat_indices_kept // n_time_bins
        time_bin_indices = flat_indices_kept % n_time_bins
        n_pts = X_traj.shape[0]
        print(f"  --drop-empty-time-bins: {n_dropped} leere Zeitbins entfernt, {n_pts} Punkte für UMAP.")
    else:
        X_traj = X_flat
        n_pts = n_pts_full
        sample_index_per_point = np.arange(n_pts) // n_time_bins
        time_bin_indices = np.arange(n_pts) % n_time_bins
    labels_traj = labels[sample_index_per_point]
    if speaker_ids is not None:
        speaker_ids_traj = speaker_ids[sample_index_per_point]
        genders_traj = genders[sample_index_per_point]
    else:
        speaker_ids_traj = None
        genders_traj = None

    print(f"UMAP (Trajektorien): {n_pts} Punkte, Dimension {X_traj.shape[1]}")
    reducer = umap.UMAP(
        n_components=args.dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        low_memory=True,
    )
    embedding_traj = reducer.fit_transform(X_traj)
    print(f"UMAP-Embedding berechnet ({args.dim}D).")

    # Embeddings in CSV speichern (vollständig, vor Downsampling)
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        col_names = [f"UMAP{i+1}" for i in range(args.dim)] + ["label", "time_bin", "sample_index"]
        if speaker_ids_traj is not None:
            col_names += ["speaker_id", "gender"]
        w = csv.writer(f)
        w.writerow(col_names)
        for i in range(embedding_traj.shape[0]):
            row = embedding_traj[i].tolist() + [
                int(labels_traj[i]), int(time_bin_indices[i]), int(sample_index_per_point[i])
            ]
            if speaker_ids_traj is not None:
                row += [int(speaker_ids_traj[i]), str(genders_traj[i])]
            w.writerow(row)
    print(f"Embeddings gespeichert: {csv_path}")

    # Einzel-Sample-Plot: eine Trajektorie nach Zeitbin, Titel = Label, Speaker ID, Gender
    n_total_samples = int(np.max(sample_index_per_point)) + 1
    sample_idx = max(0, min(args.sample_index, n_total_samples - 1))
    mask_one = sample_index_per_point == sample_idx
    if np.any(mask_one):
        emb_one = embedding_traj[mask_one]
        t_one = time_bin_indices[mask_one]
        label_one = int(labels_traj[mask_one][0])
        sid_one = int(speaker_ids_traj[mask_one][0]) if speaker_ids_traj is not None else None
        gender_one = str(genders_traj[mask_one][0]).strip() if genders_traj is not None else None
        title_parts = [f"Label {label_one}", f"Speaker ID {sid_one}" if sid_one is not None else "Speaker ID —", gender_one if gender_one else "Gender —"]
        single_title = "  |  ".join(title_parts)
        fig_one = plt.figure(figsize=(8, 6))
        if use_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax_one = fig_one.add_subplot(111, projection="3d")
            sc = ax_one.scatter(emb_one[:, 0], emb_one[:, 1], emb_one[:, 2], c=t_one, cmap="viridis", alpha=0.8, s=25, vmin=0, vmax=n_time_bins - 1)
            ax_one.set_zlabel("UMAP 3", fontsize=11)
        else:
            ax_one = fig_one.add_subplot(111)
            sc = ax_one.scatter(emb_one[:, 0], emb_one[:, 1], c=t_one, cmap="viridis", alpha=0.8, s=25, vmin=0, vmax=n_time_bins - 1)
        plt.colorbar(sc, ax=ax_one, shrink=0.7, label="Time bin")
        ax_one.set_xlabel("UMAP 1", fontsize=11)
        ax_one.set_ylabel("UMAP 2", fontsize=11)
        ax_one.set_title(single_title, fontsize=12, fontweight="bold")
        ax_one.grid(True, alpha=0.3)
        single_path = os.path.join(out_dir, f"{base_name}_single_sample{sample_idx}.png")
        plt.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Einzel-Sample-Plot gespeichert: {single_path}")

    # Downsampling für Plot: höchstens max_points Punkte (reproduzierbar mit seed)
    n_pts_orig = embedding_traj.shape[0]
    if n_pts_orig > args.max_points:
        rng = np.random.default_rng(args.seed)
        plot_idx = rng.choice(n_pts_orig, size=args.max_points, replace=False)
        plot_idx = np.sort(plot_idx)  # optional: gleiche Reihenfolge
        embedding_traj = embedding_traj[plot_idx]
        labels_traj = labels_traj[plot_idx]
        time_bin_indices = time_bin_indices[plot_idx]
        if speaker_ids_traj is not None:
            speaker_ids_traj = speaker_ids_traj[plot_idx]
            genders_traj = genders_traj[plot_idx]
        n_pts = args.max_points
        print(f"Für Plot auf {n_pts} Punkte reduziert (von {n_pts_orig}).")
    else:
        n_pts = n_pts_orig

    # Vier Plots im 2×2-Gitter: (1,1)=Label, (1,2)=Speaker, (2,1)=Gender, (2,2)=Trajektorien
    n_plots = 4
    if use_3d:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(n_plots)]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    def do_scatter(ax, x, y, z_or_none, c, label, **kwargs):
        if use_3d:
            ax.scatter(x, y, z_or_none, c=c, label=label, alpha=0.4, s=18, **kwargs)
        else:
            ax.scatter(x, y, c=c, label=label, alpha=0.4, s=22, edgecolors="none", **kwargs)

    # (1) Färbung nach Label (Trajektorien-Punkte)
    colors = get_colors_for_labels(10)
    ax = axes[0]
    for c in range(10):
        mask = labels_traj == c
        n_c = np.sum(mask)
        do_scatter(
            ax,
            embedding_traj[mask, 0], embedding_traj[mask, 1], embedding_traj[mask, 2] if use_3d else None,
            c=[colors[c]], label=f"{c} (n={n_c})",
        )
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    if use_3d:
        ax.set_zlabel("UMAP 3", fontsize=11)
    ax.set_title("By Label (digit 0–9)", fontsize=12)
    ax.legend(loc="best", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Färbung nach Speaker-ID (Trajektorien-Punkte)
    ax = axes[1]
    if speaker_ids_traj is not None:
        unique_speakers = np.unique(speaker_ids_traj)
        n_speakers = len(unique_speakers)
        cmap = get_speaker_high_contrast_cmap(max(n_speakers, 1))
        for i, sid in enumerate(unique_speakers):
            mask = speaker_ids_traj == sid
            n_s = np.sum(mask)
            do_scatter(
                ax,
                embedding_traj[mask, 0], embedding_traj[mask, 1], embedding_traj[mask, 2] if use_3d else None,
                c=[cmap(i / max(n_speakers - 1, 1))], label=f"Spk {int(sid)} (n={n_s})",
            )
        ax.legend(loc="best", ncol=2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Speaker/Gender\nnicht verfügbar", ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    if use_3d:
        ax.set_zlabel("UMAP 3", fontsize=11)
    ax.set_title("By Speaker ID", fontsize=12)
    ax.grid(True, alpha=0.3)

    # (3) Färbung nach Geschlecht (Trajektorien-Punkte)
    ax = axes[2]
    if genders_traj is not None:
        for color, lbl in zip(["#1f77b4", "#d62728"], ["male", "female"]):  # Blau = Male, Rot = Female
            mask = np.array([str(gg).lower().strip() == lbl for gg in genders_traj])
            if np.any(mask):
                n_g = np.sum(mask)
                do_scatter(
                    ax,
                    embedding_traj[mask, 0], embedding_traj[mask, 1], embedding_traj[mask, 2] if use_3d else None,
                    c=color, label=f"{lbl.capitalize()} (n={n_g})",
                )
        ax.legend(loc="best", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Speaker/Gender\nnicht verfügbar", ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    if use_3d:
        ax.set_zlabel("UMAP 3", fontsize=11)
    ax.set_title("By Speaker Gender", fontsize=12)
    ax.grid(True, alpha=0.3)

    # (4) Trajektorien: Punkte nach Zeitbin eingefärbt (gleiches UMAP wie Plots 1–3)
    ax = axes[3]
    vmin_t, vmax_t = 0, n_time_bins - 1
    if use_3d:
        sc = ax.scatter(
            embedding_traj[:, 0], embedding_traj[:, 1], embedding_traj[:, 2],
            c=time_bin_indices, cmap="viridis", alpha=0.35, s=8, rasterized=True,
            vmin=vmin_t, vmax=vmax_t,
        )
    else:
        sc = ax.scatter(
            embedding_traj[:, 0], embedding_traj[:, 1],
            c=time_bin_indices, cmap="viridis", alpha=0.35, s=8, rasterized=True,
            vmin=vmin_t, vmax=vmax_t,
        )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Time bin", fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    if use_3d:
        ax.set_zlabel("UMAP 3", fontsize=11)
    ax.set_title(f"By time bin", fontsize=11)
    ax.grid(True, alpha=0.3)

    dim_label = "3D" if use_3d else "2D"
    fig.suptitle(f"SHD Training Input — UMAP {dim_label}", fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot gespeichert: {out_path}")

    # Bei 3D: Zusätzliche Figur mit 4 Blickwinkeln (2×2) desselben Zeitbin-Plots
    if use_3d:
        from mpl_toolkits.mplot3d import Axes3D
        view_angles = [(25, 0), (25, 90), (25, 180), (25, 270)]  # elev, azim
        fig2 = plt.figure(figsize=(12, 10))
        vmin_t, vmax_t = 0, n_time_bins - 1
        for idx, (elev, azim) in enumerate(view_angles):
            ax = fig2.add_subplot(2, 2, idx + 1, projection="3d")
            sc = ax.scatter(
                embedding_traj[:, 0], embedding_traj[:, 1], embedding_traj[:, 2],
                c=time_bin_indices, cmap="viridis", alpha=0.35, s=8, rasterized=True,
                vmin=vmin_t, vmax=vmax_t,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("UMAP 1", fontsize=10)
            ax.set_ylabel("UMAP 2", fontsize=10)
            ax.set_zlabel("UMAP 3", fontsize=10)
            ax.set_title(f"By time bin (elev={elev}°, azim={azim}°)", fontsize=10)
            ax.grid(True, alpha=0.3)
        fig2.colorbar(sc, ax=fig2.axes[0], shrink=0.6, label="Time bin")
        fig2.suptitle(f"SHD Input UMAP 3D — 4 Blickwinkel", fontsize=12, y=1.02)
        plt.tight_layout()
        views_path = os.path.join(out_dir, f"{base_name}_views.png")
        plt.savefig(views_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"4-Ansichten-Plot gespeichert: {views_path}")


if __name__ == "__main__":
    main()

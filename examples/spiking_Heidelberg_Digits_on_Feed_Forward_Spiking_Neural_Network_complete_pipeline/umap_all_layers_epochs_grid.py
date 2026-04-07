"""
UMAP-Grid: Zeilen = Epoch 1, 3, 10; Spalten = Input LIF, hidden lif1, hidden lif2, Readout LIF.
Erstellt vier PNGs: Färbung nach Label, Speaker, Gender, Timebin.
Mit Achsenbeschriftungen und -zahlen in allen Plots, Legende unter allen Plots.
"""

import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm
from torch.utils.data import DataLoader
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
from manifolduntanglinganalysis.preprocessing.dataloader import H5Dataset, TransformedDataset

from umap_visualization_trajectories import (
    create_umap_embedding,
    get_high_contrast_colormap,
    get_speaker_high_contrast_colors,
    sort_key,
    load_speaker_gender_from_activity_log_h5,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Grid: Zeilen = Epochs, Spalten = Layer
EPOCHS = [1, 3, 10]
COLUMN_LAYERS = ["lif0", "lif1", "lif2", "lif3"]
COLUMN_TITLES = ["Input LIF", "hidden lif1", "hidden lif2", "Readout LIF"]

# Max. Punkte pro Subplot (Downsampling wie in umap_input_data_shd.py)
MAX_POINTS_PLOT = 10000
DOWNSCALE_SEED = 42


def _downsample_indices(n, max_points=MAX_POINTS_PLOT, seed=DOWNSCALE_SEED):
    """Reproduzierbare Indizes für Plot-Downsampling (höchstens max_points von n)."""
    if n <= max_points:
        return np.arange(n, dtype=np.intp)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _make_grid_figure(n_rows, n_cols):
    """Erstellt Figur und GridSpec für das 3x4-Grid mit Kopfzeile/Spalte.
    Größere Abstände und linke Spalte, damit Achsenbeschriftungen und Epoch-Labels nicht überlappen.
    """
    fig = plt.figure(figsize=(4.2 * n_cols + 1.2, 4.2 * n_rows + 0.8))
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols + 1,
        figure=fig,
        height_ratios=[0.14] + [1] * n_rows,
        width_ratios=[0.28] + [1] * n_cols,  # mehr Platz links für "Epoch X"
        hspace=0.28, wspace=0.32,  # mehr Abstand zwischen Subplots
    )
    fig.add_subplot(gs[0, 0]).axis("off")
    for col, title in enumerate(COLUMN_TITLES):
        ax = fig.add_subplot(gs[0, col + 1])
        ax.axis("off")
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=11, fontweight="bold")
    for row, epoch in enumerate(EPOCHS):
        ax = fig.add_subplot(gs[row + 1, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, f"Epoch {epoch}", ha="center", va="center", fontsize=11, fontweight="bold")
    return fig, gs


def main(time_bins=None, n_neighbors=None, plots_dir=None):
    """
    time_bins: Anzahl Zeitbins im Preprocessing (Default: 10).
    n_neighbors: UMAP n_neighbors (Default: 30).
    plots_dir: Verzeichnis für die 4 PNGs (Default: PROJECT_ROOT/data/plots).
    """
    activity_logs_path = os.environ.get(
        "ACTIVITY_LOGS_PATH",
        os.path.join(PROJECT_ROOT, "data", "activity_logs_feed_forward"),
    )
    if plots_dir is None:
        plots_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if time_bins is None:
        time_bins = 10
    if n_neighbors is None:
        n_neighbors = 30

    if not os.path.isdir(activity_logs_path):
        print(f"Activity-Logs-Verzeichnis nicht gefunden: {activity_logs_path}")
        return

    log_files = sorted(
        [f for f in os.listdir(activity_logs_path) if f.endswith(".h5")],
        key=sort_key,
    )
    path_by_epoch_layer = {}
    layer_neurons = {}
    for f in log_files:
        m = re.match(r"epoch_(\d+)_(\w+)_spk_events\.h5", f)
        if not m:
            continue
        epoch, layer = int(m.group(1)), m.group(2)
        path = os.path.join(activity_logs_path, f)
        path_by_epoch_layer[(epoch, layer)] = path
        if layer not in layer_neurons:
            try:
                with h5py.File(path, "r") as h5:
                    layer_neurons[layer] = int(h5.attrs["num_features"])
            except Exception:
                layer_neurons[layer] = 350

    n_rows, n_cols = len(EPOCHS), len(COLUMN_LAYERS)
    max_samples_per_class = 80

    # Einmal UMAP pro (epoch, layer) berechnen und cachen
    cache = {}
    for row, epoch in enumerate(EPOCHS):
        for col, layer in enumerate(COLUMN_LAYERS):
            key = (epoch, layer)
            if key not in path_by_epoch_layer:
                continue
            activity_log_path = path_by_epoch_layer[key]
            num_neurons = layer_neurons.get(layer, 350)
            transform = datatransforms.get_activity_logpreprocessing(
                num_neurons=num_neurons,
                fixed_duration=80,
                n_time_bins=time_bins,
            )
            try:
                h5_dataset = H5Dataset(activity_log_path)
                transformed_dataset = TransformedDataset(h5_dataset, transform)
                dataloader_obj = DataLoader(
                    transformed_dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=0,
                )
            except Exception as e:
                print(f"  Fehler laden Epoch {epoch} Layer {layer}: {e}")
                continue
            speaker_ids_epoch, genders_epoch = load_speaker_gender_from_activity_log_h5(
                activity_log_path, PROJECT_ROOT
            )
            try:
                embedding, labels, sample_indices, time_bin_indices, original_sample_indices = create_umap_embedding(
                    dataloader_obj,
                    max_samples_per_class=max_samples_per_class,
                    speaker_ids=speaker_ids_epoch,
                    min_samples_per_speaker=10,
                    n_neighbors=n_neighbors,
                )
            except Exception as e:
                print(f"  Fehler UMAP Epoch {epoch} Layer {layer}: {e}")
                continue
            valid = ~np.any(np.isnan(embedding), axis=1) & ~np.any(np.isinf(embedding), axis=1)
            cache[key] = {
                "embedding": embedding[valid],
                "labels": labels[valid],
                "sample_indices": sample_indices[valid],
                "time_bin_indices": time_bin_indices[valid],
                "original_sample_indices": original_sample_indices,
                "speaker_ids_epoch": speaker_ids_epoch,
                "genders_epoch": genders_epoch,
            }

    custom_cmap = get_high_contrast_colormap(n_colors=20)

    # ---- Grid 1: Färbung nach Label ----
    fig, gs = _make_grid_figure(n_rows, n_cols)
    label_counts = {l: 0 for l in range(10)}

    for row, epoch in enumerate(EPOCHS):
        for col, layer in enumerate(COLUMN_LAYERS):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            key = (epoch, layer)
            if key not in cache:
                ax.axis("off")
                continue
            data = cache[key]
            emb, lab = data["embedding"].copy(), data["labels"].copy()
            # Downsampling: max. MAX_POINTS_PLOT pro Subplot (reproduzierbar pro Zelle)
            plot_idx = _downsample_indices(len(emb), seed=DOWNSCALE_SEED + epoch * 100 + col)
            emb = emb[plot_idx]
            lab = lab[plot_idx]
            # Legende: Anzahl nur aus einem Subplot (erster = Epoch 1, Input LIF)
            if row == 0 and col == 0:
                for l in range(10):
                    label_counts[l] = int(np.sum(lab == l))
            ax.scatter(
                emb[:, 0], emb[:, 1],
                c=lab, cmap=custom_cmap, alpha=0.6, s=12, edgecolors="none",
                vmin=0, vmax=19,
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)

    unique_labels = list(range(10))
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=custom_cmap(l / 19.0),
            markersize=10, markeredgecolor="k", markeredgewidth=0.5,
            label=f"Label {int(l)} (n={label_counts.get(int(l), 0)})",
        )
        for l in unique_labels
    ]
    plt.tight_layout(rect=[0, 0.09, 1, 1])  # mehr Platz unten für Legende
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=10,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        title="Amount of drawed points per Subplot",
    )
    out_labels = os.path.join(plots_dir, "umap_labels_all_layers_epochs_grid.png")
    plt.savefig(out_labels, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Grid (Labels) gespeichert: {out_labels}")

    # ---- Grid 2: Färbung nach Speaker (einheitliche Farben über alle Zellen) ----
    all_speaker_ids_ordered = []
    seen = set()
    for key in sorted(cache.keys()):
        sid = cache[key].get("speaker_ids_epoch")
        if sid is None:
            continue
        for s in np.unique(sid):
            s = int(s)
            if s not in seen:
                seen.add(s)
                all_speaker_ids_ordered.append(s)
    n_speakers = max(len(all_speaker_ids_ordered), 1)
    speaker_cmap = get_high_contrast_colormap(n_colors=max(n_speakers, 20))
    sid_to_idx = {s: i for i, s in enumerate(all_speaker_ids_ordered)}
    speaker_colors = get_speaker_high_contrast_colors(n_speakers) if n_speakers <= 20 else None
    # Punkte pro Speaker zählen (über alle Zellen, auf Basis der tatsächlich geplotteten Punkte)
    speaker_point_counts = {int(s): 0 for s in all_speaker_ids_ordered}

    fig, gs = _make_grid_figure(n_rows, n_cols)
    for row, epoch in enumerate(EPOCHS):
        for col, layer in enumerate(COLUMN_LAYERS):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            key = (epoch, layer)
            if key not in cache:
                ax.axis("off")
                continue
            data = cache[key]
            emb = data["embedding"].copy()
            sid_epoch = data["speaker_ids_epoch"]
            orig_idx = data["original_sample_indices"]
            sample_idx = data["sample_indices"].astype(int)
            file_sample = np.asarray(orig_idx)[np.minimum(sample_idx, len(orig_idx) - 1)]
            if sid_epoch is not None and len(sid_to_idx) > 0:
                if np.max(file_sample) >= len(sid_epoch):
                    file_sample = np.minimum(file_sample, len(sid_epoch) - 1)
                sid = sid_epoch[file_sample]
                c = np.array([sid_to_idx.get(int(s), 0) for s in sid])
                vmin, vmax = 0, max(len(all_speaker_ids_ordered) - 1, 0)
                cmap_cell = speaker_cmap
            else:
                c = np.zeros(len(emb))
                vmin, vmax = 0, 1
                cmap_cell = "viridis"
            # Downsampling: max. MAX_POINTS_PLOT pro Subplot
            plot_idx = _downsample_indices(len(emb), seed=DOWNSCALE_SEED + epoch * 100 + col)
            emb = emb[plot_idx]
            c = c[plot_idx]
            # Legende: Anzahl nur aus einem Subplot (erster = Epoch 1, Input LIF)
            if row == 0 and col == 0 and sid_epoch is not None and len(sid_to_idx) > 0:
                sid_plot = sid[plot_idx]
                sid_int = np.array([int(s) for s in sid_plot])
                uniq_s, cnt_s = np.unique(sid_int, return_counts=True)
                for s_val, c_val in zip(uniq_s, cnt_s):
                    speaker_point_counts[int(s_val)] = int(c_val)
            ax.scatter(
                emb[:, 0], emb[:, 1],
                c=c, cmap=cmap_cell, alpha=0.6, s=12, edgecolors="none",
                vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)

    if all_speaker_ids_ordered:
        norm = mcolors.Normalize(vmin=0, vmax=max(len(all_speaker_ids_ordered) - 1, 1))
        cmap_leg = speaker_colors if speaker_colors is not None else speaker_cmap
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=cmap_leg(norm(i)),
                markersize=10, markeredgecolor="k", markeredgewidth=0.5,
                label=f"Spk {all_speaker_ids_ordered[i]} (n={speaker_point_counts.get(int(all_speaker_ids_ordered[i]), 0)})",
            )
            for i in range(len(all_speaker_ids_ordered))
        ]
        ncol_leg = min(15, len(handles))
        plt.tight_layout(rect=[0, 0.09, 1, 1])
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=ncol_leg,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.01),
            frameon=True,
            title="Punktanzahl in einem Subplot (Epoch 1, Input LIF)",
        )
    else:
        plt.tight_layout(rect=[0, 0.09, 1, 1])
    out_speaker = os.path.join(plots_dir, "umap_speaker_all_layers_epochs_grid.png")
    plt.savefig(out_speaker, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Grid (Speaker) gespeichert: {out_speaker}")

    # ---- Grid 3: Färbung nach Gender ----
    # Blau = Male, Rot = Female (wie in umap_visualization_trajectories)
    gender_cmap = mcolors.ListedColormap(["#1f77b4", "#d62728"])

    # Punkte pro Gender zählen
    male_points = 0
    female_points = 0

    fig, gs = _make_grid_figure(n_rows, n_cols)
    for row, epoch in enumerate(EPOCHS):
        for col, layer in enumerate(COLUMN_LAYERS):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            key = (epoch, layer)
            if key not in cache:
                ax.axis("off")
                continue
            data = cache[key]
            emb = data["embedding"].copy()
            genders_epoch = data["genders_epoch"]
            orig_idx = data["original_sample_indices"]
            sample_idx = data["sample_indices"].astype(int)
            file_sample = np.asarray(orig_idx)[np.minimum(sample_idx, len(orig_idx) - 1)]
            if genders_epoch is not None:
                if np.max(file_sample) >= len(genders_epoch):
                    file_sample = np.minimum(file_sample, len(genders_epoch) - 1)
                g = np.array([str(genders_epoch[i]).lower().strip() for i in file_sample])
                c = (g == "female").astype(int)  # 0 = male, 1 = female
                vmin, vmax = 0, 1
                cmap_cell = gender_cmap
            else:
                c = np.zeros(len(emb))
                vmin, vmax = 0, 1
                cmap_cell = "viridis"
            # Downsampling: max. MAX_POINTS_PLOT pro Subplot
            plot_idx = _downsample_indices(len(emb), seed=DOWNSCALE_SEED + epoch * 100 + col)
            emb = emb[plot_idx]
            c = c[plot_idx]
            # Legende: Anzahl nur aus einem Subplot (erster = Epoch 1, Input LIF)
            if row == 0 and col == 0 and genders_epoch is not None:
                g_down = np.array([str(genders_epoch[i]).lower().strip() for i in file_sample])[plot_idx]
                male_points = int(np.sum(g_down == "male"))
                female_points = int(np.sum(g_down == "female"))
            ax.scatter(
                emb[:, 0], emb[:, 1],
                c=c, cmap=cmap_cell, alpha=0.6, s=12, edgecolors="none",
                vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
            markersize=10, markeredgecolor="k", markeredgewidth=0.5, label=f"Male (n={male_points})",
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#d62728",
            markersize=10, markeredgecolor="k", markeredgewidth=0.5, label=f"Female (n={female_points})",
        ),
    ]
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        title="Punktanzahl in einem Subplot (Epoch 1, Input LIF)",
    )
    out_gender = os.path.join(plots_dir, "umap_gender_all_layers_epochs_grid.png")
    plt.savefig(out_gender, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Grid (Gender) gespeichert: {out_gender}")

    # ---- Grid 4: Färbung nach Timebin ----
    timebin_cmap = "viridis"
    n_timebin_legend = time_bins  # 0 .. time_bins-1

    # Punkte pro Timebin zählen
    timebin_counts = np.zeros(n_timebin_legend, dtype=int)

    fig, gs = _make_grid_figure(n_rows, n_cols)
    for row, epoch in enumerate(EPOCHS):
        for col, layer in enumerate(COLUMN_LAYERS):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            key = (epoch, layer)
            if key not in cache:
                ax.axis("off")
                continue
            data = cache[key]
            emb = data["embedding"].copy()
            tb = data["time_bin_indices"].copy()
            # Downsampling: max. MAX_POINTS_PLOT pro Subplot
            plot_idx = _downsample_indices(len(emb), seed=DOWNSCALE_SEED + epoch * 100 + col)
            emb = emb[plot_idx]
            tb = tb[plot_idx]
            # Legende: Anzahl nur aus einem Subplot (erster = Epoch 1, Input LIF)
            if row == 0 and col == 0 and len(tb):
                timebin_counts[:] = 0
                tb_int = tb.astype(int)
                bins, cnts = np.unique(tb_int, return_counts=True)
                for b, c in zip(bins, cnts):
                    if 0 <= b < n_timebin_legend:
                        timebin_counts[int(b)] = int(c)
            vmin, vmax = 0, max(int(np.max(tb)) if len(tb) else 0, 0)
            ax.scatter(
                emb[:, 0], emb[:, 1],
                c=tb, cmap=timebin_cmap, alpha=0.6, s=12, edgecolors="none",
                vmin=0, vmax=max(vmax, 1),
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)

    # Legende: Time bin 0 .. n_timebin_legend-1
    norm = mcolors.Normalize(vmin=0, vmax=max(n_timebin_legend - 1, 1))
    cmap = mpl_cm.get_cmap(timebin_cmap)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=cmap(norm(i)),
            markersize=10, markeredgecolor="k", markeredgewidth=0.5,
            label=f"Time bin {i} (n={int(timebin_counts[i])})",
        )
        for i in range(n_timebin_legend)
    ]
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(10, n_timebin_legend),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        title="Punktanzahl in einem Subplot (Epoch 1, Input LIF)",
    )
    out_timebin = os.path.join(plots_dir, "umap_timebins_all_layers_epochs_grid.png")
    plt.savefig(out_timebin, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Grid (Timebins) gespeichert: {out_timebin}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UMAP-Grid über Epochs und Layer (Labels, Speaker, Gender, Timebin).")
    parser.add_argument("--time-bins", type=int, default=10, help="Anzahl Zeitbins (Default: 10)")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors (Default: 30)")
    parser.add_argument("--out-dir", type=str, default=None, help="Unterordner unter data/plots, z.B. tb10_nn30 (Default: data/plots direkt)")
    args = parser.parse_args()
    base = os.path.join(PROJECT_ROOT, "data", "plots")
    plots_dir = os.path.join(base, args.out_dir) if args.out_dir else base
    main(time_bins=args.time_bins, n_neighbors=args.n_neighbors, plots_dir=plots_dir)

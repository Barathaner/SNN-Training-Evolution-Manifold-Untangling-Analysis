"""
UMAP plot for a single sample with 40 time bins.
Loads one sample from the activity logs (feed-forward pipeline), runs UMAP on its
40 time-step trajectory (40 points in high-dim space), and plots the 2D embedding
colored by time bin.

Usage:
  python umap_single_sample_40timebins.py [--epoch 1] [--layer lif0] [--sample-index 0] [--out ...]
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
from manifolduntanglinganalysis.preprocessing.dataloader import H5Dataset, TransformedDataset

try:
    import umap
except ImportError:
    raise SystemExit("UMAP not installed. Install with: pip install umap-learn")

N_TIME_BINS = 60


def _get_activity_log_path(project_root, epoch=1, layer="lif0"):
    activity_logs_path = os.environ.get(
        "ACTIVITY_LOGS_PATH",
        os.path.join(project_root, "data", "activity_logs_feed_forward"),
    )
    if not os.path.isdir(activity_logs_path):
        return None, None
    for f in os.listdir(activity_logs_path):
        m = re.match(r"epoch_(\d+)_(\w+)_spk_events\.h5", f)
        if m and int(m.group(1)) == epoch and m.group(2) == layer:
            return os.path.join(activity_logs_path, f), int(m.group(1))
    return None, None


def main():
    parser = argparse.ArgumentParser(description="UMAP for one sample, 40 time bins (trajectory).")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch of activity log")
    parser.add_argument("--layer", type=str, default="lif0", choices=["lif0", "lif1", "lif2", "lif3"], help="Layer name")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in the dataset (0-based)")
    parser.add_argument("--n-neighbors", type=int, default=5, help="UMAP n_neighbors (must be < 40; default 5)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    path, _ = _get_activity_log_path(project_root, args.epoch, args.layer)
    if path is None:
        print(f"No activity log found for epoch={args.epoch}, layer={args.layer}")
        return

    with h5py.File(path, "r") as h5:
        num_neurons = int(h5.attrs.get("num_features", 350))

    transform = datatransforms.get_activity_logpreprocessing(
        num_neurons=num_neurons,
        fixed_duration=80,
        n_time_bins=N_TIME_BINS,
    )
    h5_dataset = H5Dataset(path)
    transformed_dataset = TransformedDataset(h5_dataset, transform)
    n_samples = len(transformed_dataset)
    sample_idx = max(0, min(args.sample_index, n_samples - 1))

    events, label = transformed_dataset[sample_idx]
    if hasattr(events, "numpy"):
        events = events.numpy()
    events = np.asarray(events)
    # Shape (T, ...) -> (T, features)
    if events.ndim == 3:
        events = events.reshape(events.shape[0], -1)
    elif events.ndim == 4:
        events = events[:, 0, 0, :] if events.shape[1] == 1 and events.shape[2] == 1 else events.reshape(events.shape[0], -1)
    T, features = events.shape
    if T != N_TIME_BINS:
        print(f"Warning: sample has {T} time bins (expected {N_TIME_BINS}). Using n_neighbors <= {T-1}.")

    n_neighbors = min(max(2, args.n_neighbors), T - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(events)

    time_bins = np.arange(T)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=time_bins, cmap="viridis", alpha=0.9, s=40, vmin=0, vmax=T - 1)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="Time bin")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Single sample trajectory ({T} time bins)  |  Sample index {sample_idx}  |  Label {int(label)}")
    ax.grid(True, alpha=0.3)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(project_root, "data", "plots", f"umap_single_sample_40tb_epoch{args.epoch}_{args.layer}_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

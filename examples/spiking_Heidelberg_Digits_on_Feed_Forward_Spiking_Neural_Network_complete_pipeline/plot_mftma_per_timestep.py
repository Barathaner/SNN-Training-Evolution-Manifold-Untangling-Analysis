"""
Liest mftma_per_timestep.json und erstellt pro Epoche eine Figur mit allen Metriken (2×2-Subplots):
- X-Achse: Time steps, Y-Achse: Metrikwert
- Eine Linie pro Layer, Farbe von Hellgrün (erster Layer) bis Dunkellila (letzter Layer).
Speichert pro Epoche eine PNG (z. B. mftma_per_timestep_epoch1.png).
"""

import json
import math
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def light_green_to_dark_purple(n):
    """Erzeugt n Farben von Hellgrün bis Dunkellila."""
    green = mcolors.to_rgba("#90EE90")  # light green
    purple = mcolors.to_rgba("#2E0854")  # dark purple
    return [
        mcolors.to_hex(
            tuple((1 - i / max(n - 1, 1)) * g + (i / max(n - 1, 1)) * p for g, p in zip(green, purple))
        )
        for i in range(n)
    ]


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "results", "mftma_per_timestep40tb.json")
    out_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(data_path):
        print(f"Datei nicht gefunden: {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Epochs sortiert, Layers und Metriken aus erster Epoche
    epoch_keys = sorted(data.keys(), key=int)
    first_epoch = data[epoch_keys[0]]
    layer_names = sorted([k for k in first_epoch.keys() if isinstance(first_epoch[k], dict)])
    metric_names = [k for k in first_epoch[layer_names[0]].keys() if k != "timesteps"]

    colors = light_green_to_dark_purple(len(layer_names))
    n_metrics = len(metric_names)

    for epoch_key in epoch_keys:
        epoch_data = data[epoch_key]
        epoch_num = int(epoch_key)
        # Eine Figur pro Epoche: 2×2-Subplots für alle Metriken
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        for metric_idx, metric in enumerate(metric_names):
            ax = axes_flat[metric_idx]
            for layer_idx, layer in enumerate(layer_names):
                if layer not in epoch_data:
                    continue
                layer_data = epoch_data[layer]
                timesteps = layer_data.get("timesteps", [])
                values = layer_data.get(metric, [])
                ts_clean = []
                vals_clean = []
                for t, v in zip(timesteps, values):
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        ts_clean.append(t)
                        vals_clean.append(float(v))
                if ts_clean and vals_clean:
                    ax.plot(ts_clean, vals_clean, color=colors[layer_idx], label=layer, linewidth=2)
            ax.set_xlabel("Time step", fontsize=10)
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle(f"MFTMA per timestep — Epoch {epoch_num}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"mftma_per_timestep_epoch{epoch_num}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gespeichert: {out_path}")

    print(f"Fertig. Alle Plots in: {out_dir}")


if __name__ == "__main__":
    main()

"""
Sweep: UMAP-Grid für alle Kombinationen von time_bins und n_neighbors.
Speichert die 4 Plots pro Lauf in einem Ordner namens tb{time_bins}_nn{n_neighbors}.

Verwendung:
  python run_umap_grid_sweep.py

Ausgabeordner z.B.: data/plots/tb1_nn5/, data/plots/tb1_nn10/, ... data/plots/tb80_nn100/
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Sweep-Parameter (wie gewünscht)
TIME_BINS_LIST = [1, 3, 5, 8, 10, 15, 20, 40, 60, 80]
N_NEIGHBORS_LIST = [5, 10, 15, 20, 30, 40, 60, 100]

BASE_PLOTS = os.path.join(PROJECT_ROOT, "data", "plots")


def main():
    from umap_all_layers_epochs_grid import main as run_grid

    total = len(TIME_BINS_LIST) * len(N_NEIGHBORS_LIST)
    run_index = 0
    for time_bins in TIME_BINS_LIST:
        for n_neighbors in N_NEIGHBORS_LIST:
            run_index += 1
            out_subdir = f"tb{time_bins}_nn{n_neighbors}"
            plots_dir = os.path.join(BASE_PLOTS, out_subdir)
            print(f"\n[{run_index}/{total}] time_bins={time_bins}, n_neighbors={n_neighbors} -> {plots_dir}")
            try:
                run_grid(time_bins=time_bins, n_neighbors=n_neighbors, plots_dir=plots_dir)
            except Exception as e:
                print(f"  FEHLER: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nFertig. {total} Läufe (Ordner unter {BASE_PLOTS}).")


if __name__ == "__main__":
    main()

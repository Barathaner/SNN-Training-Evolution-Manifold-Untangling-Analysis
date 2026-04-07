#!/usr/bin/env python3
"""
Loss-Contour-Plot für das Spiking Neural Network (SFFNN).

Visualisiert die Loss-Landschaft in einer 2D-Ebene durch den aktuellen
Parameterpunkt. Zwei zufällige (orthonormale) Richtungen im Parameterraum
werden gewählt; für jedes Gitter (α, β) wird der Verlust θ + α*d1 + β*d2
auf einem Teil der Daten evaluiert und als Kontour/3D-Oberfläche geplottet.

Verwendung:
  python plot_loss_contours.py [--checkpoint PATH] [--grid 25] [--range 1.0] [--batches 10] [--no-3d]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Projekt-Root (ein Verzeichnis über dem Skript)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader

# snntorch für LIF-Neuronen (wie sffnn_batched)
import snntorch as snn


def _build_net_from_checkpoint(state_dict, num_steps=80, beta=0.9, device=None):
    """
    Liest die Architektur aus dem state_dict und baut ein kompatibles Net
    (350 -> H0 -> H1 -> H2 -> 10). Unterstützt sowohl die aktuelle Codebasis
    (H0=350) als auch Checkpoints mit H0=128.
    """
    # PyTorch Linear: weight shape (out_features, in_features)
    w0 = state_dict["fc0.weight"]
    w1 = state_dict["fc1.weight"]
    w2 = state_dict["fc2.weight"]
    w3 = state_dict["fc3.weight"]
    num_inputs = w0.shape[1]
    num_hidden0 = w0.shape[0]
    num_hidden1 = w1.shape[0]
    num_hidden2 = w2.shape[0]
    num_outputs = w3.shape[0]
    return FlexNet(
        num_inputs=num_inputs,
        num_hidden0=num_hidden0,
        num_hidden1=num_hidden1,
        num_hidden2=num_hidden2,
        num_outputs=num_outputs,
        num_steps=num_steps,
        beta=beta,
    ).to(device or "cpu")


class FlexNet(nn.Module):
    """SFFNN mit konfigurierbaren Layer-Größen (für Checkpoint-Kompatibilität)."""

    def __init__(self, num_inputs, num_hidden0, num_hidden1, num_hidden2, num_outputs, num_steps, beta):
        super().__init__()
        self.fc0 = nn.Linear(num_inputs, num_hidden0)
        self.lif0 = snn.Leaky(beta=beta)
        self.fc1 = nn.Linear(num_hidden0, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden2, num_outputs)
        self.lif3 = snn.Leaky(beta=beta)
        self.num_steps = num_steps

    def forward(self, x):
        B, T, _ = x.shape
        assert T == self.num_steps
        mem0 = self.lif0.reset_mem()
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        spk3_rec, mem3_rec = [], []
        for step in range(T):
            x_t = x[:, step, :]
            cur0 = self.fc0(x_t)
            spk0, mem0 = self.lif0(cur0, mem0)
            cur1 = self.fc1(spk0)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        spk3_bt = torch.stack(spk3_rec, dim=0).permute(1, 0, 2)
        mem3_bt = torch.stack(mem3_rec, dim=0).permute(1, 0, 2)
        return spk3_bt, mem3_bt


def get_flattened_params(model):
    """Liefert alle Modellparameter als einen flachen Vektor und die zugehörigen Shapes."""
    shapes = []
    vec = []
    for p in model.parameters():
        shapes.append(p.shape)
        vec.append(p.data.view(-1).clone())
    return torch.cat(vec), shapes


def set_flattened_params(model, flat_params, shapes):
    """Setzt Modellparameter aus einem flachen Vektor (in-place)."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].view(p.shape))
        offset += numel


def get_random_directions(flat_params, seed=42):
    """Erzeugt zwei orthonormale Zufallsrichtungen im Parameterraum (gleiche Norm wie θ)."""
    torch.manual_seed(seed)
    n = flat_params.numel()
    # Skalierung der Richtungen: gleiche „Größenordnung“ wie die Parameter
    scale = flat_params.norm().item()
    if scale < 1e-8:
        scale = 1.0
    d1 = torch.randn(n, device=flat_params.device, dtype=flat_params.dtype)
    d1 = d1 / (d1.norm() + 1e-10) * scale
    d2 = torch.randn(n, device=flat_params.device, dtype=flat_params.dtype)
    d2 = d2 - (d2.dot(d1) / (d1.dot(d1) + 1e-10)) * d1
    d2 = d2 / (d2.norm() + 1e-10) * scale
    return d1, d2


def compute_loss_over_batches(model, dataloader, loss_fn, device, max_batches=None):
    """Berechnet den durchschnittlichen Loss über (maximal) max_batches Batches."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (events, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if events.ndim == 4:
                events = events.squeeze(2)
            events = events.to(device).float()
            labels = labels.to(device)
            spk_rec, _ = model(events)
            spike_sums = spk_rec.sum(dim=1)
            loss = loss_fn(spike_sums, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / total_samples if total_samples > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Loss-Contours für SFFNN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pfad zu gespeicherten Gewichten (z.B. models/model_export/model_weights.pth)")
    parser.add_argument("--grid", type=int, default=25, help="Gittergröße (grid x grid)")
    parser.add_argument("--range", type=float, default=1.0,
                        help="Range für α und β in Einheiten der Richtungsnorm (symmetrisch ±range)")
    parser.add_argument("--batches", type=int, default=10,
                        help="Anzahl Batches pro Gitterpunkt für Loss-Schätzung")
    parser.add_argument("--seed", type=int, default=42, help="Seed für Zufallsrichtungen")
    parser.add_argument("--no-3d", action="store_true", help="Nur 2D-Kontour, keine 3D-Ansicht")
    parser.add_argument("--out", type=str, default=None,
                        help="Ausgabepfad für Plot (Standard: plots/loss_contours.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Daten
    data_path = os.path.join(PROJECT_ROOT, "data", "input")
    transform = datatransforms.get_preprocessing(
        n_time_bins=80,
        target_neurons=350,
        original_neurons=700,
        fixed_duration=958007.0,
    )
    train_loader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10),
        data_path=data_path,
        transform=transform,
        train=True,
        batch_size=64,
    )

    # Modell: Architektur aus Checkpoint ableiten oder Standard (wie main.py: 350→350→128→64→10)
    if args.checkpoint:
        ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=device)
        state_dict = loaded.get("state_dict", loaded)  # Falls Checkpoint ein Dict mit "state_dict"-Key ist
        net = _build_net_from_checkpoint(state_dict, num_steps=80, beta=0.9, device=device)
        net.load_state_dict(state_dict, strict=True)
        print(f"Checkpoint geladen: {ckpt_path} (Architektur aus Checkpoint)")
    else:
        net = FlexNet(
            num_inputs=350,
            num_hidden0=350,
            num_hidden1=128,
            num_hidden2=64,
            num_outputs=10,
            num_steps=80,
            beta=0.9,
        ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    flat_theta, shapes = get_flattened_params(net)
    d1, d2 = get_random_directions(flat_theta, seed=args.seed)

    # Gitter
    grid_size = args.grid
    r = args.range
    alphas = np.linspace(-r, r, grid_size)
    betas = np.linspace(-r, r, grid_size)
    loss_grid = np.full((grid_size, grid_size), np.nan)

    print("Berechne Loss-Gitter (kann etwas dauern)...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            new_params = flat_theta + alpha * d1 + beta * d2
            set_flattened_params(net, new_params, shapes)
            loss_grid[i, j] = compute_loss_over_batches(
                net, train_loader, loss_fn, device, max_batches=args.batches
            )
        print(f"  Zeile {i+1}/{grid_size}")

    # Zurück zum ursprünglichen Modell
    set_flattened_params(net, flat_theta, shapes)

    # Plots
    out_path = args.out or os.path.join(PROJECT_ROOT, "plots", "loss_contours.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    Alpha, Beta = np.meshgrid(alphas, betas, indexing="ij")
    valid = np.isfinite(loss_grid)
    vmin = np.nanmin(loss_grid) if np.any(valid) else 0
    vmax = np.nanpercentile(loss_grid[valid], 95) if np.sum(valid) > 0 else 1

    if args.no_3d:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cf = ax.contourf(Alpha, Beta, loss_grid, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(Alpha, Beta, loss_grid, levels=15, colors="k", linewidths=0.3, alpha=0.5)
        ax.plot(0, 0, "r*", markersize=14, label="Aktueller Parameterpunkt (0,0)")
        ax.set_xlabel(r"$\alpha$ (Richtung 1)")
        ax.set_ylabel(r"$\beta$ (Richtung 2)")
        ax.set_title("Loss-Landschaft (2D-Schnitt)")
        ax.legend()
        plt.colorbar(cf, ax=ax, label="Loss")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        fig = plt.figure(figsize=(14, 5))
        # 2D-Kontour
        ax1 = fig.add_subplot(121)
        cf = ax1.contourf(Alpha, Beta, loss_grid, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.contour(Alpha, Beta, loss_grid, levels=15, colors="k", linewidths=0.3, alpha=0.5)
        ax1.plot(0, 0, "r*", markersize=14, label="(0,0)")
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel(r"$\beta$")
        ax1.set_title("Loss-Kontour (2D-Schnitt)")
        ax1.legend()
        plt.colorbar(cf, ax=ax1, label="Loss")

        # 3D-Surface
        ax2 = fig.add_subplot(122, projection="3d")
        surf = ax2.plot_surface(Alpha, Beta, loss_grid, cmap="viridis", alpha=0.9, antialiased=True)
        ax2.scatter([0], [0], [loss_grid[grid_size // 2, grid_size // 2]], color="red", s=80, marker="*")
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"$\beta$")
        ax2.set_zlabel("Loss")
        ax2.set_title("Loss-Oberfläche")
        fig.colorbar(surf, ax=ax2, shrink=0.5, label="Loss")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Plot gespeichert: {out_path}")


if __name__ == "__main__":
    main()

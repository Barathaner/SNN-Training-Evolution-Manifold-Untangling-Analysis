#!/usr/bin/env python3
"""
Basis-Statistik für das SHD (Spiking Heidelberg Digits) Dataset.

Berechnet:
  - Spike Counts (Sparsity): Min, Max, Mittelwert, Std der Spikes pro Sample
  - Temporal Duration: Min, Max, Mittel in ms und in Zeitbins (optional)
  - Channel Utilization: Wie viele Kanäle feuern pro Sample; Spike-Häufigkeit pro Kanal
  - Global Firing Rate: Spikes pro Sekunde / pro Zeitbin
  - Class Balance: Sample-Anzahl pro Klasse (20 Klassen: 0–9 EN/DE)
  - Empty/Dead Samples: Samples mit 0 oder sehr wenigen Spikes (Qualitätskontrolle)
  - Temporal concentration: Std der Spike-Zeiten pro Sample (global + pro Klasse)
  - Samples per speaker: Verteilung der Samples pro Sprecher (aus H5)
  - Correlation spike count vs. duration: Pearson und Spearman
  - Entropy of spike-count distribution: Entropie der Spike-Count-Verteilung

Verwendung:
  python shd_dataset_statistics.py [--data-path DIR] [--max-samples N] [--time-bin-us US] [--dead-threshold N] [--json OUT]
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tonic

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None

# SHD: 700 Kanäle, 20 Klassen (Ziffern 0–9 Englisch + Deutsch)
NUM_CHANNELS = 700
NUM_CLASSES = 20
# Zeit in SHD: Mikrosekunden (μs)
US_PER_MS = 1000.0
MS_PER_S = 1000.0


def _ensure_events_fields(events):
    """Stellt sicher, dass Events 't' und 'x' haben (Tonic-Standard)."""
    if events is None or len(events) == 0:
        return None
    if not hasattr(events.dtype, "names") or events.dtype.names is None:
        return None
    if "t" not in events.dtype.names or "x" not in events.dtype.names:
        return None
    return events


def _load_speakers_for_indices(data_path: str, train: bool, indices: np.ndarray) -> np.ndarray | None:
    """Lädt Speaker-IDs für die gegebenen Indizes aus der SHD-H5-Datei. Gibt None zurück, wenn keine H5 gefunden."""
    base = Path(data_path)
    for h5_path in [
        base / "SHD" / ("shd_train.h5" if train else "shd_test.h5"),
        base / ("shd_train.h5" if train else "shd_test.h5"),
    ]:
        if not h5_path.is_file():
            continue
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                all_speakers = f["extra"]["speaker"][:]
            return all_speakers[indices]
        except Exception:
            continue
    return None


def _entropy_of_distribution(values: np.ndarray) -> float:
    """Entropie der empirischen Verteilung (diskrete Werte). H = -sum p*log(p)."""
    if len(values) == 0:
        return 0.0
    unique, counts = np.unique(values.astype(np.int64), return_counts=True)
    p = counts / counts.sum()
    # 0*log(0) = 0
    return float(-np.sum(p * np.log(p + 1e-20)))


def compute_statistics(
    data_path: str,
    train: bool,
    max_samples: int | None = None,
    time_bin_us: float | None = None,
    dead_spike_threshold: int = 10,
) -> dict:
    """
    Berechnet alle SHD-Statistiken für einen Split (Train oder Test).

    time_bin_us: Wenn gesetzt, wird Dauer auch in „Anzahl Zeitbins“ ausgegeben (z. B. für num_steps).
    """
    dataset = tonic.datasets.SHD(save_to=data_path, train=train, transform=None)
    n_total = len(dataset)
    indices = np.arange(n_total)
    if max_samples is not None and max_samples < n_total:
        np.random.seed(42)
        indices = np.random.choice(indices, size=max_samples, replace=False)
    n_used = len(indices)

    spike_counts = []
    durations_ms = []
    durations_bins = []  # nur wenn time_bin_us gesetzt
    channels_used_per_sample = []
    channel_spike_counts = np.zeros(NUM_CHANNELS, dtype=np.int64)
    channel_spike_counts_per_class = np.zeros((NUM_CLASSES, NUM_CHANNELS), dtype=np.int64)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_duration_sec = 0.0
    total_spikes_global = 0
    empty_count = 0
    dead_count = 0  # <= dead_spike_threshold
    # Temporal concentration: Std der Spike-Zeiten pro Sample (nur bei ≥2 Spikes)
    temporal_conc_ms = []
    temporal_conc_labels = []

    for idx in indices:
        events, label = dataset[idx]
        events = _ensure_events_fields(events)
        n_spikes = len(events) if events is not None else 0

        spike_counts.append(n_spikes)
        class_counts[label] += 1

        if n_spikes == 0:
            empty_count += 1
            durations_ms.append(0.0)
            channels_used_per_sample.append(0)
            if time_bin_us is not None:
                durations_bins.append(0.0)
        elif 0 < n_spikes <= dead_spike_threshold:
            dead_count += 1
            t = events["t"]
            dur_ms = (t.max() - t.min()) / US_PER_MS
            durations_ms.append(dur_ms)
            if n_spikes >= 2:
                temporal_conc_ms.append(float(np.std(t) / US_PER_MS))
                temporal_conc_labels.append(label)
            if time_bin_us is not None and time_bin_us > 0:
                durations_bins.append((t.max() - t.min()) / time_bin_us)
            unique_x = np.unique(events["x"])
            channels_used_per_sample.append(len(unique_x))
            for ch in unique_x:
                cnt = int(np.sum(events["x"] == ch))
                channel_spike_counts[ch] += cnt
                channel_spike_counts_per_class[label, ch] += cnt
            total_spikes_global += n_spikes
            total_duration_sec += dur_ms / MS_PER_S
        else:
            t = events["t"]
            dur_ms = (t.max() - t.min()) / US_PER_MS
            durations_ms.append(dur_ms)
            if n_spikes >= 2:
                temporal_conc_ms.append(float(np.std(t) / US_PER_MS))
                temporal_conc_labels.append(label)
            if time_bin_us is not None and time_bin_us > 0:
                durations_bins.append((t.max() - t.min()) / time_bin_us)
            unique_x = np.unique(events["x"])
            channels_used_per_sample.append(len(unique_x))
            for ch in unique_x:
                cnt = int(np.sum(events["x"] == ch))
                channel_spike_counts[ch] += cnt
                channel_spike_counts_per_class[label, ch] += cnt
            total_spikes_global += n_spikes
            total_duration_sec += dur_ms / MS_PER_S

    spike_counts = np.array(spike_counts, dtype=np.float64)
    durations_ms = np.array(durations_ms)
    non_empty = spike_counts > 0
    n_non_empty = int(np.sum(non_empty))

    # Firing rate: Spikes pro Sekunde (nur über Samples mit Dauer > 0)
    if total_duration_sec > 0:
        firing_rate_per_sec = total_spikes_global / total_duration_sec
    else:
        firing_rate_per_sec = 0.0

    # Firing rate pro Zeitbin (wenn time_bin_us gesetzt)
    if time_bin_us is not None and time_bin_us > 0 and n_non_empty > 0:
        duration_bins_arr = np.array(durations_bins)
        total_bins = np.sum(duration_bins_arr[duration_bins_arr > 0])
        if total_bins > 0:
            firing_rate_per_bin = total_spikes_global / total_bins
        else:
            firing_rate_per_bin = 0.0
    else:
        total_bins = None
        firing_rate_per_bin = None

    # Channel utilization: Anteil der Kanäle, die in einem typischen Sample feuern
    channels_used_per_sample = np.array(channels_used_per_sample) if channels_used_per_sample else np.array([])

    # Top-10-Kanäle pro Klasse (nach Spike-Anzahl in dieser Klasse)
    top10_channels_per_class = []
    top10_counts_per_class = []
    for c in range(NUM_CLASSES):
        order = np.argsort(channel_spike_counts_per_class[c])[::-1][:10]
        top10_channels_per_class.append([int(ch) for ch in order])
        top10_counts_per_class.append([int(channel_spike_counts_per_class[c, ch]) for ch in order])

    # Temporal concentration (Std der Spike-Zeiten in ms, pro Sample mit ≥2 Spikes)
    temporal_conc_ms = np.array(temporal_conc_ms) if temporal_conc_ms else np.array([])
    temporal_conc_labels = np.array(temporal_conc_labels) if temporal_conc_labels else np.array([])
    temporal_concentration = {}
    if len(temporal_conc_ms) > 0:
        temporal_concentration["global_ms"] = {
            "min": float(np.min(temporal_conc_ms)),
            "max": float(np.max(temporal_conc_ms)),
            "mean": float(np.mean(temporal_conc_ms)),
            "std": float(np.std(temporal_conc_ms)),
            "n_samples_with_ge2_spikes": int(len(temporal_conc_ms)),
        }
        per_class_mean = []
        for c in range(NUM_CLASSES):
            mask = temporal_conc_labels == c
            if np.any(mask):
                per_class_mean.append(float(np.mean(temporal_conc_ms[mask])))
            else:
                per_class_mean.append(None)
        temporal_concentration["per_class_mean_ms"] = per_class_mean
    else:
        temporal_concentration["global_ms"] = None
        temporal_concentration["per_class_mean_ms"] = [None] * NUM_CLASSES

    # Samples per speaker (aus H5)
    speakers = _load_speakers_for_indices(data_path, train, indices)
    samples_per_speaker = {}
    if speakers is not None:
        unique_s, counts_s = np.unique(speakers, return_counts=True)
        samples_per_speaker["speaker_ids"] = [int(s) for s in unique_s]
        samples_per_speaker["counts"] = [int(c) for c in counts_s]
        samples_per_speaker["min"] = int(np.min(counts_s))
        samples_per_speaker["max"] = int(np.max(counts_s))
        samples_per_speaker["mean"] = float(np.mean(counts_s))
        samples_per_speaker["std"] = float(np.std(counts_s))
        samples_per_speaker["n_speakers"] = len(unique_s)
    else:
        samples_per_speaker["available"] = False

    # Correlation: spike count vs. duration (nur Samples mit Spikes und Dauer > 0)
    valid_corr = non_empty & (durations_ms > 0)
    if np.sum(valid_corr) >= 2 and pearsonr is not None and spearmanr is not None:
        r_pearson, p_pearson = pearsonr(spike_counts[valid_corr], durations_ms[valid_corr])
        r_spearman, p_spearman = spearmanr(spike_counts[valid_corr], durations_ms[valid_corr])
        correlation_spike_count_duration = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "n_samples_used": int(np.sum(valid_corr)),
        }
    else:
        correlation_spike_count_duration = {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "n_samples_used": int(np.sum(valid_corr)) if np.sum(valid_corr) >= 2 else 0,
        }

    # Entropy of spike-count distribution
    entropy_spike_count = _entropy_of_distribution(spike_counts)

    result = {
        "split": "train" if train else "test",
        "n_samples_total": n_total,
        "n_samples_used": n_used,
        "spike_counts_sparsity": {
            "min": int(np.min(spike_counts)) if len(spike_counts) else 0,
            "max": int(np.max(spike_counts)) if len(spike_counts) else 0,
            "mean": float(np.mean(spike_counts)) if len(spike_counts) else 0.0,
            "std": float(np.std(spike_counts)) if len(spike_counts) else 0.0,
        },
        "temporal_duration_ms": {
            "min": float(np.min(durations_ms)) if len(durations_ms) else 0.0,
            "max": float(np.max(durations_ms)) if len(durations_ms) else 0.0,
            "mean": float(np.mean(durations_ms)) if len(durations_ms) else 0.0,
        },
        "channel_utilization": {
            "channels_used_per_sample_min": int(np.min(channels_used_per_sample)) if len(channels_used_per_sample) else 0,
            "channels_used_per_sample_max": int(np.max(channels_used_per_sample)) if len(channels_used_per_sample) else 0,
            "channels_used_per_sample_mean": float(np.mean(channels_used_per_sample)) if len(channels_used_per_sample) else 0.0,
            "channels_used_per_sample_std": float(np.std(channels_used_per_sample)) if len(channels_used_per_sample) else 0.0,
            "channel_spike_counts_min": int(np.min(channel_spike_counts)) if np.any(channel_spike_counts) else 0,
            "channel_spike_counts_max": int(np.max(channel_spike_counts)),
            "channel_spike_counts_mean": float(np.mean(channel_spike_counts)),
            "channel_spike_counts_std": float(np.std(channel_spike_counts)),
            "channels_with_at_least_one_spike": int(np.sum(channel_spike_counts > 0)),
        },
        "top10_channels_per_class": {
            "channel_ids": top10_channels_per_class,
            "channel_counts": top10_counts_per_class,
        },
        "global_firing_rate": {
            "spikes_per_second": firing_rate_per_sec,
            "spikes_per_time_bin": firing_rate_per_bin,
        },
        "class_balance": {
            "per_class_counts": [int(c) for c in class_counts],
            "min_count": int(np.min(class_counts)) if np.any(class_counts) else 0,
            "max_count": int(np.max(class_counts)),
            "mean_count": float(np.mean(class_counts)),
            "std_count": float(np.std(class_counts)),
            "is_balanced": bool(np.std(class_counts) < 1e-6) if n_used > 0 else False,
        },
        "quality_control": {
            "empty_samples_count": int(empty_count),
            "empty_samples_ratio": float(empty_count / n_used) if n_used else 0.0,
            "dead_samples_count": int(dead_count),
            "dead_samples_ratio": float(dead_count / n_used) if n_used else 0.0,
            "dead_threshold_spikes": dead_spike_threshold,
        },
        "temporal_concentration": temporal_concentration,
        "samples_per_speaker": samples_per_speaker,
        "correlation_spike_count_duration": correlation_spike_count_duration,
        "entropy_spike_count_distribution": entropy_spike_count,
    }

    if time_bin_us is not None and time_bin_us > 0 and len(durations_bins):
        result["temporal_duration_time_bins"] = {
            "time_bin_us": time_bin_us,
            "min": float(np.min(durations_bins)),
            "max": float(np.max(durations_bins)),
            "mean": float(np.mean(durations_bins)),
        }

    return result


def print_report(stats_train: dict, stats_test: dict, time_bin_us: float | None):
    """Gibt einen lesbaren Report in die Konsole aus."""
    def section(title):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    for name, stats in [("TRAIN", stats_train), ("TEST", stats_test)]:
        section(f"SHD Dataset — {name} (n={stats['n_samples_used']} von {stats['n_samples_total']})")

        print("\n📊 Spike Counts (Sparsity)")
        s = stats["spike_counts_sparsity"]
        print(f"   Min: {s['min']}, Max: {s['max']}, Mean: {s['mean']:.2f}, Std: {s['std']:.2f}")

        print("\n⏱️  Temporal Duration (ms)")
        t = stats["temporal_duration_ms"]
        print(f"   Min: {t['min']:.2f} ms, Max: {t['max']:.2f} ms, Mean: {t['mean']:.2f} ms")
        if "temporal_duration_time_bins" in stats:
            tb = stats["temporal_duration_time_bins"]
            print(f"   (Time bins, {tb['time_bin_us']} µs/bin): Min: {tb['min']:.1f}, Max: {tb['max']:.1f}, Mean: {tb['mean']:.1f}")

        print("\n📡 Channel Utilization (Spatial Distribution)")
        c = stats["channel_utilization"]
        print(f"   Channels used per sample — Min: {c['channels_used_per_sample_min']}, Max: {c['channels_used_per_sample_max']}, Mean: {c['channels_used_per_sample_mean']:.1f} (±{c['channels_used_per_sample_std']:.1f})")
        print(f"   Spikes per channel (global) — Min: {c['channel_spike_counts_min']}, Max: {c['channel_spike_counts_max']}, Mean: {c['channel_spike_counts_mean']:.1f}")
        print(f"   Channels with ≥1 spike: {c['channels_with_at_least_one_spike']} / {NUM_CHANNELS}")

        top10 = stats.get("top10_channels_per_class", {})
        if top10 and top10.get("channel_ids"):
            print("\n📌 Top 10 Channels per Class (channel IDs and spike counts)")
            ids_per_class = top10["channel_ids"]
            counts_per_class = top10.get("channel_counts", [])
            for cls in range(min(20, len(ids_per_class))):
                ch_ids = ids_per_class[cls]
                ch_cnt = counts_per_class[cls] if cls < len(counts_per_class) else []
                if ch_cnt:
                    pairs = " ".join(f"{ch}({n})" for ch, n in zip(ch_ids, ch_cnt))
                else:
                    pairs = " ".join(str(ch) for ch in ch_ids)
                print(f"   Class {cls:2d}: {pairs}")

        print("\n🔥 Global Firing Rate")
        fr = stats["global_firing_rate"]
        print(f"   Spikes per second: {fr['spikes_per_second']:.2f}")
        if fr.get("spikes_per_time_bin") is not None:
            print(f"   Spikes per time bin: {fr['spikes_per_time_bin']:.2f}")

        print("\n📋 Class Balance (20 classes: digits 0–9 EN/DE)")
        cb = stats["class_balance"]
        print(f"   Per class: {cb['per_class_counts']}")
        print(f"   Min: {cb['min_count']}, Max: {cb['max_count']}, Mean: {cb['mean_count']:.1f}, Std: {cb['std_count']:.1f}")
        print(f"   Perfectly balanced: {cb['is_balanced']}")

        print("\n⚠️  Quality Control (Empty / Dead Samples)")
        qc = stats["quality_control"]
        print(f"   Empty (0 spikes): {qc['empty_samples_count']} ({qc['empty_samples_ratio']*100:.2f}%)")
        print(f"   Dead (≤{qc['dead_threshold_spikes']} spikes): {qc['dead_samples_count']} ({qc['dead_samples_ratio']*100:.2f}%)")

        # Temporal concentration (Std der Spike-Zeiten pro Sample, in ms)
        tc = stats.get("temporal_concentration", {})
        if tc and tc.get("global_ms"):
            g = tc["global_ms"]
            print("\n⏱️  Temporal Concentration (Std der Spike-Zeiten pro Sample, ms)")
            print(f"   Min: {g['min']:.2f}, Max: {g['max']:.2f}, Mean: {g['mean']:.2f}, Std: {g['std']:.2f} (n≥2 Spikes: {g['n_samples_with_ge2_spikes']})")
            pc = tc.get("per_class_mean_ms")
            if pc and any(x is not None for x in pc):
                valid = [f"{i}:{v:.1f}" for i, v in enumerate(pc) if v is not None]
                print(f"   Per-class mean (ms): {', '.join(valid[:10])}{'...' if len(valid) > 10 else ''}")
        else:
            print("\n⏱️  Temporal Concentration: (keine Samples mit ≥2 Spikes)")

        # Samples per speaker
        sps = stats.get("samples_per_speaker", {})
        if not sps.get("speaker_ids"):
            print("\n👥 Samples per Speaker: (H5 extra/speaker nicht verfügbar)")
        else:
            print("\n👥 Samples per Speaker")
            print(f"   Speakers: {sps['n_speakers']}, Min: {sps['min']}, Max: {sps['max']}, Mean: {sps['mean']:.1f}, Std: {sps['std']:.1f}")
            print(f"   Counts per speaker: {sps['counts'][:15]}{'...' if len(sps.get('counts', [])) > 15 else ''}")

        # Correlation spike count vs duration
        corr = stats.get("correlation_spike_count_duration", {})
        if corr.get("pearson_r") is not None:
            print("\n📈 Correlation (Spike Count vs. Duration)")
            print(f"   Pearson  r = {corr['pearson_r']:.4f}, p = {corr['pearson_p']:.4e} (n = {corr['n_samples_used']})")
            print(f"   Spearman r = {corr['spearman_r']:.4f}, p = {corr['spearman_p']:.4e}")
        else:
            n_used_corr = corr.get("n_samples_used", 0)
            if n_used_corr < 2:
                print("\n📈 Correlation (Spike Count vs. Duration): (zu wenige gültige Samples)")
            else:
                print("\n📈 Correlation (Spike Count vs. Duration): (scipy nicht installiert)")

        # Entropy of spike-count distribution
        ent = stats.get("entropy_spike_count_distribution")
        if ent is not None:
            print(f"\n📊 Entropy of spike-count distribution: {ent:.4f} (nats)")

    print()


def main():
    parser = argparse.ArgumentParser(description="SHD Dataset — Basis-Statistik")
    parser.add_argument("--data-path", type=str, default=None,
                        help=f"Pfad zum SHD-Datensatz (Standard: {{project}}/data/input)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max. Samples pro Split (Standard: alle)")
    parser.add_argument("--time-bin-us", type=float, default=None,
                        help="Zeitbin in µs (z. B. 11975 für ~80 Bins bei 958007 µs) — für Dauer in Zeitbins und Firing/Bin")
    parser.add_argument("--dead-threshold", type=int, default=10,
                        help="Spike-Grenze für 'Dead Sample' (Standard: 10)")
    parser.add_argument("--json", type=str, default=None,
                        help="Ausgabe als JSON-Datei")
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(PROJECT_ROOT, "data", "input")
    if not os.path.isdir(data_path):
        print(f"Data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Optional: Zeitbin aus fester Dauer und 80 Bins (wie im Projekt)
    if args.time_bin_us is None:
        fixed_duration_us = 958_007.0
        n_bins = 80
        time_bin_us = fixed_duration_us / n_bins
        print(f"Using time_bin_us = {time_bin_us:.1f} (fixed_duration={fixed_duration_us} µs, n_bins={n_bins})")
    else:
        time_bin_us = args.time_bin_us

    stats_train = compute_statistics(
        data_path=data_path,
        train=True,
        max_samples=args.max_samples,
        time_bin_us=time_bin_us,
        dead_spike_threshold=args.dead_threshold,
    )
    stats_test = compute_statistics(
        data_path=data_path,
        train=False,
        max_samples=args.max_samples,
        time_bin_us=time_bin_us,
        dead_spike_threshold=args.dead_threshold,
    )

    print_report(stats_train, stats_test, time_bin_us)

    if args.json:
        out = {
            "time_bin_us": time_bin_us,
            "num_channels": NUM_CHANNELS,
            "num_classes": NUM_CLASSES,
            "train": stats_train,
            "test": stats_test,
        }
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"JSON gespeichert: {args.json}")


if __name__ == "__main__":
    main()

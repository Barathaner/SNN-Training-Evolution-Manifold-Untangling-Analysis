"""
Einfaches Skript zum Erstellen von UMAP-Visualisierungen aus Activity Logs.
Erstellt für jeden Layer ein Bild mit 20 Subplots (eine pro Epoche).
"""

import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
from manifolduntanglinganalysis.preprocessing.dataloader import H5Dataset, TransformedDataset


def get_high_contrast_colormap(n_colors=20):
    """
    Erstellt eine hochkontrastige Colormap mit n_colors.
    Verwendet eine Kombination aus verschiedenen Farbpaletten für maximale Unterscheidbarkeit.
    """
    # Basis-Farben mit hohem Kontrast
    base_colors = [
        '#1f77b4',  # Blau
        '#ff7f0e',  # Orange
        '#2ca02c',  # Grün
        '#d62728',  # Rot
        '#9467bd',  # Lila
        '#8c564b',  # Braun
        '#e377c2',  # Pink
        '#7f7f7f',  # Grau
        '#bcbd22',  # Oliv
        '#17becf',  # Cyan
        '#aec7e8',  # Hellblau
        '#ffbb78',  # Hellorange
        '#98df8a',  # Hellgrün
        '#ff9896',  # Hellrot
        '#c5b0d5',  # Helllila
        '#c49c94',  # Hellbraun
        '#f7b6d3',  # Hellpink
        '#c7c7c7',  # Hellgrau
        '#dbdb8d',  # Helloliv
        '#9edae5',  # Hellcyan
    ]
    
    # Wenn mehr Farben benötigt werden, generiere zusätzliche
    if n_colors > len(base_colors):
        # Generiere zusätzliche Farben mit hohem Kontrast
        import colorsys
        additional_colors = []
        for i in range(len(base_colors), n_colors):
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio für gleichmäßige Verteilung
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.5 + (i % 2) * 0.3
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            additional_colors.append(mcolors.rgb2hex(rgb))
        base_colors.extend(additional_colors)
    
    # Erstelle ListedColormap
    return mcolors.ListedColormap(base_colors[:n_colors])


def get_speaker_high_contrast_colors(n_colors):
    """
    Maximally distinct colors for speakers (keine ähnlichen Blautöne).
    Gleichmäßig über den Farbkreis (HSV-Hue) verteilt, hohe Sättigung.
    """
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio für gute Verteilung
        saturation = 0.85
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(mcolors.rgb2hex(rgb))
    return mcolors.ListedColormap(colors[:n_colors])


def sort_key(filename):
    """Sortiere Activity Logs nach Epoche und Layer."""
    match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', filename)
    if match:
        epoch = int(match.group(1))
        layer = match.group(2)
        return (epoch, layer)
    return (999, 'zzz')


def load_speaker_gender_from_activity_log_h5(activity_log_path, project_root=None):
    """
    Liest Speaker-IDs (und optional Gender) direkt aus der Activity-Log-H5.
    Die Activity Logs speichern bei include_metadata=True pro Sample Metadaten
    (z. B. metadata/sample_0.attrs['speakers']).
    Returns:
        speaker_ids: (N,) int, N = Anzahl Samples in der H5; oder None falls nicht vorhanden.
        genders: (N,) str, sofern project_root für SHD-Gender-Lookup angegeben; sonst None.
    """
    try:
        with h5py.File(activity_log_path, 'r') as f:
            if 'metadata' not in f:
                return None, None
            meta = f['metadata']
            sample_keys = sorted([k for k in meta.keys() if k.startswith('sample_')],
                                key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            if not sample_keys:
                return None, None
            speaker_ids = []
            for k in sample_keys:
                grp = meta[k]
                # ActivityMonitor speichert pro Sample z. B. 'speakers' (vom SHDMetadataExtractor)
                sid = grp.attrs.get('speakers', grp.attrs.get('speaker', None))
                if sid is None:
                    return None, None
                speaker_ids.append(int(sid))
            speaker_ids = np.array(speaker_ids)
    except Exception:
        return None, None
    # Gender: aus SHD extra/meta_info/gender (Index = Speaker-ID)
    genders = None
    if project_root:
        data_path = os.path.join(project_root, "data", "input")
        for h5_path in [Path(data_path) / "SHD" / "shd_train.h5", Path(data_path) / "shd_train.h5"]:
            if not h5_path.is_file():
                continue
            try:
                with h5py.File(h5_path, 'r') as f:
                    gender_raw = f["extra"]["meta_info"]["gender"][:]
                speaker_genders = [g.decode("utf-8") if hasattr(g, "decode") else str(g) for g in gender_raw]
                genders = np.array([speaker_genders[int(sid)] for sid in speaker_ids])
                break
            except Exception:
                continue
    return speaker_ids, genders


def load_speaker_gender_for_shd_order(project_root, max_sample_index_plus_one, label_range=None):
    """
    Fallback: Lädt Speaker/Gender aus SHD-H5 in Trainings-Reihenfolge (wenn Activity-Log keine Metadaten hat).
    """
    if label_range is None:
        label_range = list(range(10))
    label_set = set(label_range)
    data_path = os.path.join(project_root, "data", "input")
    base = Path(data_path)
    for h5_path in [base / "SHD" / "shd_train.h5", base / "shd_train.h5", Path(project_root) / "data" / "input" / "shd_train.h5"]:
        if not h5_path.is_file():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                all_speakers = f["extra"]["speaker"][:]
                gender_raw = f["extra"]["meta_info"]["gender"][:]
                labels = f["labels"][:]
            speaker_genders = [g.decode("utf-8") if hasattr(g, "decode") else str(g) for g in gender_raw]
            filtered_indices = np.array([i for i in range(len(labels)) if int(labels[i]) in label_set])
            indices = filtered_indices[:max_sample_index_plus_one]
            speaker_ids = all_speakers[indices]
            genders = np.array([speaker_genders[int(sid)] for sid in speaker_ids])
            return speaker_ids, genders
        except Exception:
            continue
    return None, None


def create_umap_embedding(dataloader, n_neighbors=20, min_dist=0.25, random_state=42, max_samples_per_class=None,
                          speaker_ids=None, min_samples_per_speaker=5):
    """
    Erstellt UMAP-Embedding für einen DataLoader.
    
    Args:
        dataloader: DataLoader mit Daten
        n_neighbors: Anzahl Nachbarn für UMAP
        min_dist: Mindestabstand für UMAP
        random_state: Random Seed
        max_samples_per_class: Max. Samples pro Klasse (bei speaker_ids=None) bzw. max. Samples pro Speaker (bei speaker-balanced).
        speaker_ids: Optional (N,) Speaker-ID pro Sample in Dataloader-Reihenfolge; für speaker-balanced Sampling.
        min_samples_per_speaker: Nur Speaker mit mindestens so vielen Samples (nur bei speaker-balanced).
    
    Returns:
        embedding, labels, sample_indices, time_bin_indices, original_sample_indices
    """
    # Sammle Daten und Labels auf Sample-Ebene (bevor Trajektorien geflacht werden)
    all_data = []
    all_sample_labels = []
    all_original_indices = []  # Index in der H5-Datei (sample_0, sample_1, ...)
    
    np.random.seed(random_state)
    global_idx = 0
    for events, labels in dataloader:
        if events.ndim == 4:
            events = events.squeeze(2)
        events_np = events.numpy() if isinstance(events, torch.Tensor) else events
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        batch_size, T, features = events_np.shape
        for i in range(batch_size):
            all_data.append(events_np[i])
            all_sample_labels.append(labels_np[i])
            all_original_indices.append(global_idx)
            global_idx += 1
    
    # Sampling: speaker-balanced (min 5 pro Speaker, max max_samples_per_class pro Speaker) oder pro Klasse
    if max_samples_per_class is not None:
        if speaker_ids is not None and len(speaker_ids) > 0:
            # Speaker-balanced: mind. min_samples_per_speaker, max. max_samples_per_class pro Speaker
            all_speaker_ids = np.array(
                [speaker_ids[i] if i < len(speaker_ids) else -1 for i in all_original_indices],
                dtype=np.int64
            )
            by_speaker = defaultdict(list)
            for i in range(len(all_original_indices)):
                by_speaker[all_speaker_ids[i]].append(i)
            selected_sample_indices = []
            for sid, indices in by_speaker.items():
                if len(indices) >= min_samples_per_speaker:
                    take = min(len(indices), max_samples_per_class)
                    chosen = np.random.choice(indices, size=take, replace=False)
                    selected_sample_indices.extend(chosen)
            selected_sample_indices = sorted(selected_sample_indices)
            all_data = [all_data[i] for i in selected_sample_indices]
            all_sample_labels = [all_sample_labels[i] for i in selected_sample_indices]
            all_original_indices = [all_original_indices[i] for i in selected_sample_indices]
            n_speakers = sum(1 for indices in by_speaker.values() if len(indices) >= min_samples_per_speaker)
            print(f"      Speaker-balanced: min {min_samples_per_speaker}, max {max_samples_per_class} pro Speaker → {len(selected_sample_indices)} Samples ({n_speakers} Speaker)")
        else:
            # Fallback: genau max_samples_per_class Samples pro Klasse
            unique_labels = np.unique(all_sample_labels)
            selected_sample_indices = []
            for label in unique_labels:
                label_sample_indices = [i for i, lbl in enumerate(all_sample_labels) if lbl == label]
                if len(label_sample_indices) >= max_samples_per_class:
                    selected_label_indices = np.random.choice(
                        label_sample_indices, size=max_samples_per_class, replace=False
                    )
                    selected_sample_indices.extend(selected_label_indices)
            all_data = [all_data[i] for i in selected_sample_indices]
            all_sample_labels = [all_sample_labels[i] for i in selected_sample_indices]
            all_original_indices = [all_original_indices[i] for i in selected_sample_indices]
            print(f"      Genau {max_samples_per_class} Samples pro Klasse: {len(selected_sample_indices)} Samples gesamt")
    
    original_sample_indices = np.array(all_original_indices)  # (n_selected,) = H5 sample_0, sample_1, ...
    
    # Jeder Zeitpunkt (Zeitbin) wird ein Datenpunkt
    X_list = []
    labels_list = []
    sample_indices_list = []
    time_bin_indices_list = []
    for sample_idx, (sample_data, sample_label) in enumerate(zip(all_data, all_sample_labels)):
        T, features = sample_data.shape
        X_list.append(sample_data)
        labels_list.append(np.repeat(sample_label, T))
        sample_indices_list.append(np.repeat(sample_idx, T))
        time_bin_indices_list.append(np.arange(T))
    
    X = np.concatenate(X_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    sample_indices = np.concatenate(sample_indices_list, axis=0)
    time_bin_indices = np.concatenate(time_bin_indices_list, axis=0)
    
    print(f"      Gesamt: {len(X)} Datenpunkte (aus {len(all_data)} Samples, je {all_data[0].shape[0]} Zeitbins)")
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        low_memory=True,
        n_jobs=1,  # bei random_state erzwingt UMAP ohnehin 1 Thread; unterdrückt Warnung
    )
    embedding = reducer.fit_transform(X)
    
    return embedding, labels, sample_indices, time_bin_indices, original_sample_indices


def create_umap_plot_for_layer(layer_name, activity_log_paths, output_path, num_neurons, max_samples_per_class=10,
                               color_by='label', project_root=None):
    """
    Erstellt UMAP-Visualisierung für einen Layer mit allen Epochen als Subplots.
    color_by: 'label' | 'speaker' | 'gender' | 'timebin' – Färbung; jede Variante in separater Datei.
    """
    print(f"📂 Layer: {layer_name}, Färbung: {color_by}")
    print(f"   Epochen: {len(activity_log_paths)}")
    
    activity_log_transform = datatransforms.get_activity_logpreprocessing(
        num_neurons=num_neurons,
        fixed_duration=80,
        n_time_bins=10
    )
    
    n_epochs = len(activity_log_paths)
    n_cols = 5
    n_rows = int(np.ceil(n_epochs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    custom_cmap_labels = get_high_contrast_colormap(n_colors=20)
    all_labels = None
    max_sample_idx_seen = -1
    speaker_ids_global = None
    genders_global = None
    last_epoch_labels = None
    last_epoch_speaker_ids = None
    last_epoch_genders = None
    
    for idx, activity_log_path in enumerate(activity_log_paths):
        match = re.match(r'.*epoch_(\d+)_', activity_log_path)
        epoch = int(match.group(1)) if match else idx + 1
        print(f"   Verarbeite Epoche {epoch}...")
        
        try:
            with h5py.File(activity_log_path, 'r') as f:
                if 'time_steps' in f.attrs:
                    pass  # optional debug
        except Exception:
            pass
        
        # Speaker-IDs nur aus Activity-Log (Reihenfolge = sample_0, sample_1, …) für speaker-balanced Sampling
        speaker_ids_epoch, genders_epoch = load_speaker_gender_from_activity_log_h5(activity_log_path, project_root)
        speaker_ids_for_sampling = speaker_ids_epoch  # nur Activity-Log-Reihenfolge für Sampling
        if speaker_ids_epoch is None and project_root:
            speaker_ids_global, genders_global = load_speaker_gender_for_shd_order(project_root, 10000)
        
        h5_dataset = H5Dataset(activity_log_path)
        transformed_dataset = TransformedDataset(h5_dataset, activity_log_transform)
        dataloader_obj = DataLoader(transformed_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        try:
            embedding, labels, sample_indices, time_bin_indices, original_sample_indices = create_umap_embedding(
                dataloader_obj, max_samples_per_class=max_samples_per_class, speaker_ids=speaker_ids_for_sampling,
                min_samples_per_speaker=5,
            )
            if all_labels is None:
                all_labels = labels
            max_sample_idx_seen = max(max_sample_idx_seen, int(np.max(sample_indices)))
            # Pro Punkt: Index in der H5-Datei (für Speaker/Gender aus Activity-Log)
            file_sample_per_point = original_sample_indices[sample_indices.astype(int)]
            
            valid_mask = ~np.any(np.isnan(embedding), axis=1) & ~np.any(np.isinf(embedding), axis=1)
            if not np.all(valid_mask):
                embedding = embedding[valid_mask]
                labels = labels[valid_mask]
                sample_indices = sample_indices[valid_mask]
                time_bin_indices = time_bin_indices[valid_mask]
                file_sample_per_point = file_sample_per_point[valid_mask]
            
            # Farbvektor je nach color_by
            if color_by == 'label':
                c = labels
                vmin, vmax = 0, 19
                cmap = custom_cmap_labels
            elif color_by == 'speaker':
                # Bevorzugt: Speaker aus der Activity-Log-H5 (pro Epoche); Index = H5 sample_0, sample_1, ...
                speaker_ids_epoch, genders_epoch = load_speaker_gender_from_activity_log_h5(
                    activity_log_path, project_root
                )
                if speaker_ids_epoch is None and (speaker_ids_global is None and project_root):
                    speaker_ids_global, genders_global = load_speaker_gender_for_shd_order(project_root, 10000)
                use_speakers = speaker_ids_epoch if speaker_ids_epoch is not None else speaker_ids_global
                if use_speakers is not None:
                    if speaker_ids_epoch is not None:
                        speaker_ids_global = speaker_ids_epoch
                        genders_global = genders_epoch
                    # Bei Daten aus Activity-Log: file_sample_per_point; sonst sample_indices
                    idx_for_speaker = file_sample_per_point if speaker_ids_epoch is not None else sample_indices.astype(int)
                    if np.max(idx_for_speaker) >= len(use_speakers):
                        idx_for_speaker = np.minimum(idx_for_speaker, len(use_speakers) - 1)
                    sid = use_speakers[idx_for_speaker]
                    uniq = np.unique(sid)
                    sid_ord = np.searchsorted(uniq, sid)
                    c = sid_ord
                    vmin, vmax = 0, max(len(uniq) - 1, 0)
                    cmap = get_high_contrast_colormap(n_colors=max(len(uniq), 20))  # wie bei Labels
                else:
                    c = np.zeros(len(embedding))
                    vmin, vmax = 0, 1
                    cmap = 'viridis'
            elif color_by == 'gender':
                _, genders_epoch = load_speaker_gender_from_activity_log_h5(activity_log_path, project_root)
                if genders_epoch is None and genders_global is None and project_root:
                    if speaker_ids_global is None:
                        speaker_ids_global, genders_global = load_speaker_gender_for_shd_order(project_root, 10000)
                    else:
                        _, genders_global = load_speaker_gender_for_shd_order(project_root, 10000)
                use_genders = genders_epoch if genders_epoch is not None else genders_global
                if use_genders is not None:
                    if genders_epoch is not None:
                        genders_global = genders_epoch
                    idx_for_gender = file_sample_per_point if genders_epoch is not None else sample_indices.astype(int)
                    if np.max(idx_for_gender) >= len(use_genders):
                        idx_for_gender = np.minimum(idx_for_gender, len(use_genders) - 1)
                    g = np.array([str(use_genders[i]).lower().strip() for i in idx_for_gender])
                    c = (g == 'female').astype(int)
                    vmin, vmax = 0, 1
                    cmap = mcolors.ListedColormap(['#1f77b4', '#d62728'])  # Blau = Male, Rot = Female
                    last_epoch_genders = g.copy()
                else:
                    c = np.zeros(len(embedding))
                    vmin, vmax = 0, 1
                    cmap = 'viridis'
            else:  # timebin
                c = time_bin_indices
                vmin, vmax = 0, int(np.max(time_bin_indices))
                cmap = 'viridis'
            
            ax = axes[idx]
            is_3d = embedding.shape[1] == 3
            if is_3d:
                ax.remove()
                ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
                sc = ax.scatter(
                    embedding[:, 0], embedding[:, 1], embedding[:, 2],
                    c=c, cmap=cmap, alpha=0.6, s=20, edgecolors='k', linewidths=0.5,
                    vmin=vmin, vmax=vmax
                )
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.set_zlabel('UMAP 3', fontsize=8)
            else:
                sc = ax.scatter(
                    embedding[:, 0], embedding[:, 1],
                    c=c, cmap=cmap, alpha=0.6, s=20, edgecolors='k', linewidths=0.5,
                    vmin=vmin, vmax=vmax
                )
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            if color_by == 'timebin':
                plt.colorbar(sc, ax=ax, shrink=0.6, label='Time bin')
            ax.set_title(f'Epoch {epoch}', fontsize=10, fontweight='bold')
            
        except Exception as e:
            print(f"   ⚠️  Fehler bei Epoche {epoch}: {e}")
            axes[idx].axis('off')
            axes[idx].set_title(f'Epoch {epoch} (Error)', fontsize=10)
            continue
    
    for idx in range(n_epochs, len(axes)):
        axes[idx].axis('off')
    
    # Legende für label / speaker / gender (immer anzeigen; bei vollem Gitter unter der Figur)
    if color_by == 'label' and all_labels is not None:
        unique_labels = sorted(np.unique(all_labels))
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap_labels(l / 19.0),
                             markersize=10, markeredgecolor='k', markeredgewidth=0.5, label=f'Label {int(l)}')
                  for l in unique_labels]
        if n_epochs < len(axes):
            legend_ax = axes[-1]
            legend_ax.axis('off')
            legend_ax.legend(handles=handles, loc='center', ncol=2, fontsize=8, title='Labels', title_fontsize=10, framealpha=0.9)
        else:
            fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    elif color_by == 'speaker' and speaker_ids_global is not None:
        end = min(max_sample_idx_seen + 1, len(speaker_ids_global))
        uniq_s = np.unique(speaker_ids_global[: end]) if end > 0 else np.array([])
        if len(uniq_s) > 0:
            # Gleiche Hochkontrast-Palette wie bei Labels (mind. 20 Farben)
            n_s = max(len(uniq_s), 20)
            cmap_s = get_high_contrast_colormap(n_colors=n_s)
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_s(i / max(len(uniq_s) - 1, 1)),
                                   markersize=10, markeredgecolor='k', markeredgewidth=0.5, label=f'Spk {int(uniq_s[i])}')
                      for i in range(min(len(uniq_s), 20))]
            if n_epochs < len(axes):
                legend_ax = axes[-1]
                legend_ax.axis('off')
                legend_ax.legend(handles=handles, loc='center', ncol=2, fontsize=7, title='Speaker', title_fontsize=10, framealpha=0.9)
            else:
                fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=7, bbox_to_anchor=(0.5, -0.02), title='Speaker')
    elif color_by == 'gender':
        if last_epoch_genders is not None:
            g_flat = np.array([str(x).lower().strip() for x in last_epoch_genders])
            n_male = int(np.sum(g_flat == 'male'))
            n_female = int(np.sum(g_flat == 'female'))
        else:
            n_male = n_female = 0
        handles_gender = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, markeredgecolor='k', label=f'Male (n={n_male})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, markeredgecolor='k', label=f'Female (n={n_female})'),
        ]
        if n_epochs < len(axes):
            legend_ax = axes[-1]
            legend_ax.axis('off')
            legend_ax.legend(handles=handles_gender, loc='center', fontsize=10, framealpha=0.9)
        else:
            fig.legend(handles=handles_gender, loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle(f'UMAP - {layer_name} (by {color_by})', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot gespeichert: {output_path}\n")
    return fig


if __name__ == "__main__":
    import sys
    
    # Projekt-Root bestimmen
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    activity_logs_path = os.path.join(project_root, "data", "activity_logs_feed_forward")
    plots_dir = os.path.join(project_root, "data", "plots")
    
    # Prüfe ob Activity Logs Verzeichnis existiert
    if not os.path.exists(activity_logs_path):
        print(f"❌ Activity Logs Verzeichnis nicht gefunden: {activity_logs_path}")
        sys.exit(1)
    
    # Lade alle Activity Logs
    activity_logs = [f for f in os.listdir(activity_logs_path) if f.endswith('.h5')]
    
    if len(activity_logs) == 0:
        print(f"❌ Keine Activity Logs gefunden in {activity_logs_path}")
        sys.exit(1)
    
    # Sortiere Activity Logs
    activity_logs = sorted(activity_logs, key=sort_key)
    
    # Gruppiere nach Layer
    layer_groups = {}
    layer_neurons = {}
    
    for activity_log in activity_logs:
        match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', activity_log)
        if not match:
            continue
        
        layer = match.group(2)
        activity_log_path = os.path.join(activity_logs_path, activity_log)
        
        # Lese Anzahl Neuronen (einmal pro Layer)
        if layer not in layer_neurons:
            try:
                with h5py.File(activity_log_path, 'r') as f:
                    layer_neurons[layer] = int(f.attrs['num_features'])
            except Exception as e:
                print(f"⚠️  Fehler beim Lesen von {activity_log}: {e}")
                continue
        
        # Füge zu Layer-Gruppe hinzu
        if layer not in layer_groups:
            layer_groups[layer] = []
        layer_groups[layer].append(activity_log_path)
    
    # Sortiere Activity Logs innerhalb jedes Layers nach Epoche
    for layer in layer_groups:
        layer_groups[layer] = sorted(layer_groups[layer], key=lambda x: int(re.search(r'epoch_(\d+)_', x).group(1)))
    
    print(f"📊 Gefunden: {len(activity_logs)} Activity Logs")
    print(f"📊 Layer: {list(layer_groups.keys())}\n")
    
    # Konfiguration: Exakt so viele Samples pro Klasse (Klassen mit weniger entfallen)
    max_samples_per_class = 40  # Kann hier geändert werden
    
    # Pro Layer: 4 separate Dateien (by label, speaker, gender, timebin)
    color_variants = [
        ('label', 'labels'),
        ('speaker', 'speaker'),
        ('gender', 'gender'),
        ('timebin', 'timebin'),
    ]
    
    for layer_name, activity_log_paths in layer_groups.items():
        if layer_name not in layer_neurons:
            print(f"⚠️  Überspringe {layer_name}: Keine Neuron-Information verfügbar")
            continue
        
        for color_by, suffix in color_variants:
            output_path = os.path.join(plots_dir, f"{layer_name}_all_epochs_umap_{suffix}.png")
            try:
                create_umap_plot_for_layer(
                    layer_name=layer_name,
                    activity_log_paths=activity_log_paths,
                    output_path=output_path,
                    num_neurons=layer_neurons[layer_name],
                    max_samples_per_class=max_samples_per_class,
                    color_by=color_by,
                    project_root=project_root,
                )
            except Exception as e:
                print(f"❌ Fehler bei Layer {layer_name}, Färbung {color_by}: {e}\n")
                import traceback
                traceback.print_exc()
                continue
    
    print("✅ Alle UMAP-Visualisierungen erstellt!")

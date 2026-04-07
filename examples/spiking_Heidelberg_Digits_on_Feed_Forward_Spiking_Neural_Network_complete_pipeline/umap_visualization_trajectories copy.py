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


def sort_key(filename):
    """Sortiere Activity Logs nach Epoche und Layer."""
    match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', filename)
    if match:
        epoch = int(match.group(1))
        layer = match.group(2)
        return (epoch, layer)
    return (999, 'zzz')


def create_umap_embedding(dataloader, n_neighbors=20, min_dist=0.5, random_state=42, max_samples_per_class=None):
    """
    Erstellt UMAP-Embedding für einen DataLoader.
    
    Args:
        dataloader: DataLoader mit Daten
        n_neighbors: Anzahl Nachbarn für UMAP
        min_dist: Mindestabstand für UMAP
        random_state: Random Seed
        max_samples_per_class: Maximale Anzahl Samples pro Klasse (None = alle)
                              Filtert auf Sample-Ebene, behält alle Zeitbins pro Sample
    
    Returns:
        embedding: UMAP-Embedding (N, 2 oder 3)
        labels: Labels für jeden Datenpunkt
    """
    # Sammle Daten und Labels auf Sample-Ebene (bevor Trajektorien geflacht werden)
    all_data = []
    all_labels = []
    all_sample_labels = []  # Label für jedes Sample
    
    np.random.seed(random_state)
    
    for events, labels in dataloader:
        if events.ndim == 4:
            events = events.squeeze(2)
        
        events_np = events.numpy() if isinstance(events, torch.Tensor) else events
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        batch_size, T, features = events_np.shape
        
        # Speichere jedes Sample separat mit seinem Label
        for i in range(batch_size):
            all_data.append(events_np[i])  # Shape: (T, features)
            all_sample_labels.append(labels_np[i])
    
    # Filtere auf max_samples_per_class pro Klasse (auf Sample-Ebene)
    if max_samples_per_class is not None:
        unique_labels = np.unique(all_sample_labels)
        selected_sample_indices = []
        
        for label in unique_labels:
            label_sample_indices = [i for i, lbl in enumerate(all_sample_labels) if lbl == label]
            if len(label_sample_indices) > max_samples_per_class:
                # Zufällig max_samples_per_class Samples auswählen
                selected_label_indices = np.random.choice(
                    label_sample_indices, 
                    size=max_samples_per_class, 
                    replace=False
                )
                selected_sample_indices.extend(selected_label_indices)
            else:
                selected_sample_indices.extend(label_sample_indices)
        
        # Wähle nur die ausgewählten Samples
        all_data = [all_data[i] for i in selected_sample_indices]
        all_sample_labels = [all_sample_labels[i] for i in selected_sample_indices]
        
        print(f"      Gefiltert: {len(selected_sample_indices)} Samples ({max_samples_per_class} pro Klasse)")
    
    # Jeder Zeitpunkt (Zeitbin) wird ein Datenpunkt
    # Jeder Timestep × Sample = ein Datenpunkt
    X_list = []
    labels_list = []
    
    for sample_data, sample_label in zip(all_data, all_sample_labels):
        T, features = sample_data.shape
        # Jeder Timestep wird ein Datenpunkt
        X_list.append(sample_data)  # (T, features)
        # Label für jeden Timestep wiederholen (gleiches Label für alle Zeitbins eines Samples)
        labels_list.append(np.repeat(sample_label, T))
    
    X = np.concatenate(X_list, axis=0)  # (N_samples * T, features)
    labels = np.concatenate(labels_list, axis=0)  # (N_samples * T,)
    
    print(f"      Gesamt: {len(X)} Datenpunkte (aus {len(all_data)} Samples, je {all_data[0].shape[0]} Zeitbins)")
    
    # Erstelle UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        low_memory=True
    )
    
    embedding = reducer.fit_transform(X)
    
    return embedding, labels


def create_umap_plot_for_layer(layer_name, activity_log_paths, output_path, num_neurons, max_samples_per_class=10):
    """
    Erstellt UMAP-Visualisierung für einen Layer mit allen Epochen als Subplots.
    
    Args:
        layer_name: Name des Layers (z.B. 'lif0')
        activity_log_paths: Liste von Pfaden zu Activity Log Dateien für diesen Layer (sortiert nach Epoche)
        output_path: Pfad zum Speichern des Plots
        num_neurons: Anzahl der Neuronen
        max_samples_per_class: Maximale Anzahl Samples pro Klasse (default: 10)
    """
    print(f"📂 Layer: {layer_name}")
    print(f"   Epochen: {len(activity_log_paths)}")
    
    # Erstelle Transform
    activity_log_transform = datatransforms.get_activity_logpreprocessing(
        num_neurons=num_neurons,
        fixed_duration=80,
        n_time_bins=10
    )
    
    # Erstelle hochkontrastige Colormap für 20 Labels
    custom_cmap = get_high_contrast_colormap(n_colors=20)
    
    # Erstelle Figure mit Subplots (4x5 für 20 Epochen)
    n_epochs = len(activity_log_paths)
    n_cols = 5
    n_rows = int(np.ceil(n_epochs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Sammle alle Embeddings und Labels für gemeinsame Legende
    all_labels = None
    
    # Verarbeite jede Epoche
    for idx, activity_log_path in enumerate(activity_log_paths):
        # Extrahiere Epoche aus Dateinamen
        match = re.match(r'.*epoch_(\d+)_', activity_log_path)
        epoch = int(match.group(1)) if match else idx + 1
        
        print(f"   Verarbeite Epoche {epoch}...")
        
        # Prüfe Anzahl Zeitbins vor Transformation
        try:
            with h5py.File(activity_log_path, 'r') as f:
                # Prüfe zuerst Metadaten für time_steps
                if 'time_steps' in f.attrs:
                    time_steps = int(f.attrs['time_steps'])
                    print(f"      Zeitbins vor Transformation (aus Metadaten): {time_steps}")
                elif 'events' in f:
                    events_group = f['events']
                    sample_keys = sorted(events_group.keys())
                    if len(sample_keys) > 0:
                        # Prüfe mehrere Samples für Zeitbins
                        all_times = []
                        for sample_key in sample_keys[:min(10, len(sample_keys))]:  # Prüfe bis zu 10 Samples
                            sample = events_group[sample_key]
                            if 't' in sample.dtype.names and len(sample) > 0:
                                times = sample['t']
                                all_times.extend(times)
                        
                        if len(all_times) > 0:
                            # Maximaler Zeitstempel + 1 = Anzahl Zeitbins (da 0-indexiert)
                            max_time = np.max(all_times)
                            # Oder zähle eindeutige Zeitstempel über alle Samples
                            unique_times = len(np.unique(all_times))
                            print(f"      Zeitbins vor Transformation: max_time+1={max_time+1}, unique_times={unique_times} (über {min(10, len(sample_keys))} Samples)")
                        else:
                            print(f"      Zeitbins vor Transformation: 0 (keine Events gefunden)")
                    else:
                        print(f"      Zeitbins vor Transformation: Keine Samples gefunden")
                else:
                    print(f"      Zeitbins vor Transformation: Keine Events-Gruppe gefunden")
        except Exception as e:
            print(f"      ⚠️  Konnte Zeitbins vor Transformation nicht lesen: {e}")
        
        # Lade Activity Log
        h5_dataset = H5Dataset(activity_log_path)
        transformed_dataset = TransformedDataset(h5_dataset, activity_log_transform)
        dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        # Erstelle UMAP-Embedding
        try:
            embedding, labels = create_umap_embedding(dataloader, max_samples_per_class=max_samples_per_class)
            
            # Speichere Labels für Legende (einmal)
            if all_labels is None:
                all_labels = labels
            
            # Plot auf Subplot
            ax = axes[idx]
            
            # Prüfe auf NaN oder extreme Werte
            valid_mask = ~np.any(np.isnan(embedding), axis=1) & ~np.any(np.isinf(embedding), axis=1)
            if not np.all(valid_mask):
                embedding = embedding[valid_mask]
                labels = labels[valid_mask]
            
            # Prüfe ob 3D oder 2D
            is_3d = embedding.shape[1] == 3
            
            if is_3d:
                # 3D Plot
                ax.remove()
                ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
                scatter = ax.scatter(
                    embedding[:, 0], 
                    embedding[:, 1], 
                    embedding[:, 2],
                    c=labels, 
                    cmap=custom_cmap,
                    alpha=0.6, 
                    s=20,
                    edgecolors='k', 
                    linewidths=0.5,
                    vmin=0,
                    vmax=19
                )
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.set_zlabel('UMAP 3', fontsize=8)
            else:
                # 2D Plot
                scatter = ax.scatter(
                    embedding[:, 0], 
                    embedding[:, 1], 
                    c=labels, 
                    cmap=custom_cmap,
                    alpha=0.6, 
                    s=20,
                    edgecolors='k', 
                    linewidths=0.5,
                    vmin=0,
                    vmax=19
                )
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            ax.set_title(f'Epoch {epoch}', fontsize=10, fontweight='bold')
            
        except Exception as e:
            print(f"   ⚠️  Fehler bei Epoche {epoch}: {e}")
            axes[idx].axis('off')
            axes[idx].set_title(f'Epoch {epoch} (Error)', fontsize=10)
            continue
    
    # Verstecke leere Subplots
    for idx in range(n_epochs, len(axes)):
        axes[idx].axis('off')
    
    # Füge gemeinsame Legende hinzu (nur einmal)
    if all_labels is not None:
        # Erstelle eine unsichtbare Scatter für die Legende
        unique_labels = np.unique(all_labels)
        unique_labels = sorted(unique_labels)
        
        # Verwende den letzten Subplot für die Legende (wenn Platz vorhanden)
        if n_epochs < len(axes):
            legend_ax = axes[-1]
            legend_ax.axis('off')
            
            # Erstelle Legende mit allen Labels
            handles = []
            for label in unique_labels:
                color = custom_cmap(label / 19.0)  # Normalisiere auf [0, 1]
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        markeredgecolor='k', markeredgewidth=0.5,
                                        label=f'Label {int(label)}'))
            
            legend_ax.legend(handles=handles, loc='center', ncol=2, fontsize=8, 
                           title='Labels', title_fontsize=10, framealpha=0.9)
        else:
            # Füge Legende außerhalb der Subplots hinzu
            fig.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=custom_cmap(i / 19.0), 
                                          markersize=10, markeredgecolor='k',
                                          label=f'Label {i}') 
                               for i in unique_labels],
                      loc='lower center', ncol=10, fontsize=8, 
                      bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle(f'UMAP Visualization - {layer_name} (All Epochs)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Speichere Plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot gespeichert: {output_path}\n")
    
    return fig


if __name__ == "__main__":
    import sys
    
    # Projekt-Root bestimmen
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    activity_logs_path = os.path.join(project_root, "data", "activity_logs_ffn")
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
    
    # Konfiguration: Maximale Anzahl Samples pro Klasse
    max_samples_per_class = 50  # Kann hier geändert werden
    
    # Erstelle UMAP-Visualisierung für jeden Layer
    for layer_name, activity_log_paths in layer_groups.items():
        if layer_name not in layer_neurons:
            print(f"⚠️  Überspringe {layer_name}: Keine Neuron-Information verfügbar")
            continue
        
        output_path = os.path.join(plots_dir, f"{layer_name}_all_epochs_umap.png")
        
        try:
            create_umap_plot_for_layer(
                layer_name=layer_name,
                activity_log_paths=activity_log_paths,
                output_path=output_path,
                num_neurons=layer_neurons[layer_name],
                max_samples_per_class=max_samples_per_class
            )
        except Exception as e:
            print(f"❌ Fehler bei Layer {layer_name}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("✅ Alle UMAP-Visualisierungen erstellt!")

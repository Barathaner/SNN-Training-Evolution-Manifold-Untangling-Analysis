import tonic
from tonic.transforms import ToFrame
from tonic import DiskCachedDataset
from tonic.collation import PadTensors
from torch.utils.data import Subset, Dataset, DataLoader
from typing import Optional
import h5py
import numpy as np
from pathlib import Path
import torch

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]
        # Prüfe ob Events strukturiertes Array sind (rohe Events)
        if isinstance(events, np.ndarray) and events.dtype.names is not None:
            # Spezialbehandlung für leere Events: Compose bricht bei leeren Events ab,
            # bevor ToFrame aufgerufen wird. Daher müssen wir leere Events manuell zu Frames konvertieren.
            if len(events) == 0:
                # Extrahiere ToFrame-Parameter aus dem Transform
                to_frame_params = self._extract_to_frame_params()
                if to_frame_params:
                    # Berechne n_time_bins aus end_time und time_window
                    end_time = to_frame_params.get('end_time')
                    time_window = to_frame_params.get('time_window')
                    if end_time is not None and time_window is not None and time_window > 0:
                        n_time_bins = int(end_time / time_window)
                    else:
                        n_time_bins = 80  # Fallback
                    
                    sensor_size = to_frame_params.get('sensor_size', (128, 1, 1))
                    # ToFrame gibt Frames im Format (n_time_bins, polarity, height, width) zurück
                    # Bei time_window + end_time: n_time_bins = end_time / time_window
                    # Format: (n_time_bins, polarity, height, width) = (n_time_bins, 1, 1, num_neurons)
                    empty_frames = np.zeros((n_time_bins, sensor_size[2], sensor_size[0], sensor_size[1]), dtype=np.float32)
                    
                    # Wende restliche Transforms an (z.B. GaussianSmoothing), aber überspringe ToFrame
                    if hasattr(self.transform, 'transforms') and len(self.transform.transforms) > 1:
                        # Überspringe ToFrame (erste Transform) und wende restliche an
                        for t in self.transform.transforms[1:]:
                            empty_frames = t(empty_frames)
                    return empty_frames, label
                else:
                    # Fallback: Versuche Transform trotzdem anzuwenden
                    transformed = self.transform(events)
            else:
                # Events sind nicht leer - normaler Transform
                transformed = self.transform(events)
            
            # Prüfe ob Transform Frames zurückgibt (numpy array ohne strukturierten dtype)
            if isinstance(transformed, np.ndarray) and transformed.dtype.names is not None:
                # Strukturiertes Array (Events) - Transform hat nicht funktioniert!
                raise TypeError(f"Transform hat Events statt Frames zurückgegeben! "
                              f"Typ: {type(transformed)}, dtype: {transformed.dtype}, "
                              f"Shape: {transformed.shape if hasattr(transformed, 'shape') else 'N/A'}")
            return transformed, label
        else:
            # Events sind bereits Frames oder etwas anderes
            return self.transform(events), label
    
    def _extract_to_frame_params(self):
        """Extrahiert ToFrame-Parameter aus dem Transform für leere Events."""
        if not hasattr(self.transform, 'transforms'):
            return None
        
        for t in self.transform.transforms:
            if hasattr(t, 'sensor_size') and hasattr(t, 'time_window'):
                params = {
                    'sensor_size': t.sensor_size,
                    'time_window': t.time_window,
                    'start_time': getattr(t, 'start_time', 0.0),
                    'end_time': getattr(t, 'end_time', None),
                }
                return params
        return None

def load_filtered_shd_dataloader(
    label_range=range(0, 10),
    data_path="./data",
    transform=None,
    train=True,
    batch_size=32,
    shuffle=False,
    drop_last=True,
    num_workers=0,
    num_samples: Optional[int] = None,
):
    """
    Lädt einen gefilterten und transformierten SHD-Datensatz als DataLoader.

    Args:
        label_range (iterable): Liste oder Range der gewünschten Labels (default: 0–9)
        data_path (str): Pfad zum Datensatz
        transform (callable): Event → Tensor Transform (default: ToFrame)
        train (bool): Trainings- oder Testset
        batch_size (int): Batchgröße für DataLoader
        shuffle (bool): Zufällige Batchmischung
        drop_last (bool): Letzten Batch verwerfen, wenn zu klein
        num_workers (int): DataLoader Worker
        num_samples (int, optional): Maximale Anzahl Samples nach dem Filtern. 
                                    Wenn None, werden alle gefilterten Samples verwendet.

    Returns:
        torch.utils.data.DataLoader: Dataloader für das vorbereitete SHD-Dataset
    """
    dataset_full = tonic.datasets.SHD(save_to=data_path, train=train, transform=None)

    label_range = set(label_range)
    filtered_indices = [
        i for i in range(len(dataset_full)) if dataset_full[i][1] in label_range
    ]
    
    # Begrenze auf num_samples, falls angegeben
    original_count = len(filtered_indices)
    if num_samples is not None:
        if num_samples > original_count:
            print(f"⚠️ Warnung: num_samples ({num_samples}) > gefilterte Samples ({original_count}). Verwende alle {original_count} Samples.")
        else:
            filtered_indices = filtered_indices[:num_samples]
            print(f"📊 Begrenzt auf {num_samples} Samples (von {original_count} gefilterten)")

    if transform is None:
        transform = ToFrame(
            sensor_size=tonic.datasets.SHD.sensor_size,  # = (700,)
            n_time_bins=100  # Aktualisiert auf 1000
        )
    subset = Subset(dataset_full, filtered_indices)
    transformed_dataset = TransformedDataset(subset, transform)

    # Cache-Pfad basierend auf n_time_bins, damit unterschiedliche Konfigurationen nicht kollidieren
    cache_suffix = f"n_time_bins_{transform.n_time_bins if hasattr(transform, 'n_time_bins') else 'default'}"
    cached_dataset = DiskCachedDataset(transformed_dataset, cache_path=f'./cache/shd/{cache_suffix}')
    dataloader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=PadTensors(),
        num_workers=num_workers,
    )

    return dataloader

class H5Dataset(Dataset):
    """
    Dataset-Klasse zum Laden von H5-Dateien im Tonic-Format.
    Lädt Events und Labels ohne automatische Transformation.
    """
    def __init__(self, h5_path: str):
        """
        Initialisiert das H5-Dataset.
        
        Args:
            h5_path: Pfad zur H5-Datei oder Verzeichnis mit H5-Dateien
        """
        self.h5_path = Path(h5_path)
        self.samples = []
        
        # Finde H5-Dateien
        if self.h5_path.is_file():
            h5_files = [self.h5_path]
        elif self.h5_path.is_dir():
            h5_files = sorted(self.h5_path.glob("*.h5"))
        else:
            raise FileNotFoundError(f"H5-Pfad nicht gefunden: {h5_path}")
        
        if not h5_files:
            raise ValueError(f"Keine H5-Dateien gefunden in: {h5_path}")
        
        # Lade alle Samples aus allen H5-Dateien
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # Lade Labels (falls vorhanden)
                labels = f['labels'][:] if 'labels' in f else None
                
                # Lade Events für jeden Sample
                if 'events' not in f:
                    continue
                    
                events_group = f['events']
                for sample_key in sorted(events_group.keys()):
                    events = events_group[sample_key][:]
                    
                    # Extrahiere Sample-Index und Label
                    sample_idx = int(sample_key.split('_')[1]) if '_' in sample_key else len(self.samples)
                    label = int(labels[sample_idx]) if labels is not None and sample_idx < len(labels) else 0
                    
                    self.samples.append((events, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        events, label = self.samples[idx]
        return events, label

def load_activity_log(
    activity_log_path="./data/activity_logs",
    transform=None,
    batch_size=64,
    drop_last=True,
    num_workers=0,
    num_samples: Optional[int] = None,
):
    """
    Lädt einen gefilterten Activity-Log-Datensatz als DataLoader.
    
    Transform wird nur angewendet, wenn explizit übergeben. Ohne Transform werden
    die rohen Events zurückgegeben.

    Args:
        label_range (iterable): Liste oder Range der gewünschten Labels (default: 0–9)
        activity_log_path (str): Pfad zur H5-Datei oder Verzeichnis mit H5-Dateien
        transform (callable, optional): Event → Tensor Transform. Wenn None, werden
                                      rohe Events ohne Transformation zurückgegeben.
        batch_size (int): Batchgröße für DataLoader
        shuffle (bool): Zufällige Batchmischung
        drop_last (bool): Letzten Batch verwerfen, wenn zu klein
        num_workers (int): DataLoader Worker
        num_samples (int, optional): Maximale Anzahl Samples nach dem Filtern. 
                                    Wenn None, werden alle gefilterten Samples verwendet.

    Returns:
        torch.utils.data.DataLoader: Dataloader für das Dataset
    """
    # Lade Dataset
    dataset_full = H5Dataset(activity_log_path)
    
    # Filtere nach Labels

    
    # Begrenze auf num_samples, falls angegeben
    if num_samples is not None:
        original_count = len(filtered_indices)
        if num_samples > original_count:
            print(f"⚠️ Warnung: num_samples ({num_samples}) > gefilterte Samples ({original_count}). "
                  f"Verwende alle {original_count} Samples.")
        else:
            filtered_indices = filtered_indices[:num_samples]
            print(f"📊 Begrenzt auf {num_samples} Samples (von {original_count} gefilterten)")
    
    
    # Wende Transform nur an, wenn explizit übergeben
    if transform is not None:
        dataset = TransformedDataset(dataset_full, transform)
        # KEIN Cache für Activity Logs - verursacht Probleme mit Events vs Frames
        # Der Cache kann alte Events statt transformierte Frames enthalten
        # dataset = DiskCachedDataset(dataset, cache_path=cache_path)

    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=PadTensors(),
        num_workers=num_workers,
    )

    return dataloader



def load_shd_raw_subset(train=False, label_range=range(0, 9)):
    dataset_full = tonic.datasets.SHD(save_to="./data", train=train, transform=None)

    label_range = set(label_range)
    filtered_indices = [
        i for i in range(len(dataset_full)) if dataset_full[i][1] in label_range
    ]
    
    subset = Subset(dataset_full, filtered_indices)
    return subset
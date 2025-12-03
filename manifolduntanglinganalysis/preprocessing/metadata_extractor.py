"""
Funktionen zum Extrahieren von Metadaten aus dem SHD-Dataloader für einen Batch.
"""
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

# Import für abstrakte Basisklasse (mit verzögertem Import um zirkuläre Imports zu vermeiden)
from abc import ABC, abstractmethod

class MetadataExtractor(ABC):
    """
    Abstrakte Basisklasse für Metadaten-Extraktoren.
    Wird auch in ActivityMonitor definiert, aber hier für bessere Verfügbarkeit.
    """
    @abstractmethod
    def extract(self, dataloader: DataLoader, batch_idx: int) -> Dict[str, np.ndarray]:
        """
        Extrahiert Metadaten für einen spezifischen Batch.
        
        Args:
            dataloader: Der DataLoader
            batch_idx: Index des Batches (0-basiert)
        
        Returns:
            Dictionary mit Metadaten-Arrays (z.B. {'speakers': [...], 'sample_ids': [...]})
        """
        pass


def _get_dataset_components(dataloader):
    """
    Hilfsfunktion: Navigiert durch die verschachtelte Dataset-Struktur.
    
    Returns:
        Tuple (subset, original_dataset) mit:
        - subset: Das Subset mit filtered_indices
        - original_dataset: Das Original SHD-Dataset mit speaker
    """
    # Navigiere zum Subset (hat filtered_indices)
    subset = dataloader.dataset
    while hasattr(subset, 'dataset') and not hasattr(subset, 'indices'):
        subset = subset.dataset
    
    if not hasattr(subset, 'indices'):
        raise ValueError("Konnte Subset mit 'indices' nicht finden")
    
    # Navigiere zum Original-Dataset (hat speaker)
    original_dataset = subset.dataset
    while hasattr(original_dataset, 'dataset'):
        original_dataset = original_dataset.dataset
    
    if not hasattr(original_dataset, 'speaker'):
        raise ValueError("Konnte Original-Dataset mit 'speaker' nicht finden")
    
    return subset, original_dataset


def extract_batch_metadata(dataloader, batch_idx: int = 0) -> Dict[str, np.ndarray]:
    """
    Extrahiert Metadaten (speaker, original_sample_ids) für einen spezifischen Batch.
    
    Die Speaker-IDs werden direkt aus den original_sample_ids (gefilterten Indizes) extrahiert,
    da diese die korrespondierenden Indizes im Original-Dataset sind.
    
    Args:
        dataloader: Der DataLoader
        batch_idx: Index des Batches (0-basiert)
    
    Returns:
        Dictionary mit:
        - 'speakers': Array von Speaker-IDs für jeden Sample im Batch
        - 'original_sample_ids': Array von originalen Sample-IDs (vor Filterung)
    """
    subset, original_dataset = _get_dataset_components(dataloader)
    
    # Hole gefilterte Indizes (das sind die original_sample_ids nach Filterung)
    filtered_indices = np.array(subset.indices)
    
    # Berechne die Indizes für diesen Batch im gefilterten Subset
    batch_size = dataloader.batch_size
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(filtered_indices))
    
    # Hole die original_sample_ids für diesen Batch
    batch_filtered_indices = filtered_indices[start_idx:end_idx]
    original_sample_ids = batch_filtered_indices
    
    # Hole Speaker-IDs direkt aus dem Original-Dataset
    # Die original_sample_ids sind die korrespondierenden Indizes
    all_speakers = original_dataset.speaker
    batch_speakers = all_speakers[batch_filtered_indices]
    
    return {
        'speakers': batch_speakers,
        'original_sample_ids': original_sample_ids
    }


def extract_batch_metadata_simple(dataloader) -> Dict[str, np.ndarray]:
    """
    Vereinfachte Funktion zum Extrahieren von Metadaten für den ersten Batch.
    Wrapper um extract_batch_metadata mit batch_idx=0.
    
    Args:
        dataloader: Der DataLoader
    
    Returns:
        Dictionary mit:
        - 'speakers': Array von Speaker-IDs für jeden Sample im Batch
        - 'original_sample_ids': Array von originalen Sample-IDs (vor Filterung)
    """
    return extract_batch_metadata(dataloader, batch_idx=0)


class SHDMetadataExtractor(MetadataExtractor):
    """
    Konkrete Implementierung von MetadataExtractor für SHD-Datasets.
    """
    
    def extract(self, dataloader: DataLoader, batch_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Extrahiert Metadaten für einen spezifischen Batch aus dem SHD-Dataloader.
        
        Args:
            dataloader: Der DataLoader
            batch_idx: Index des Batches (0-basiert)
        
        Returns:
            Dictionary mit:
            - 'speakers': Array von Speaker-IDs für jeden Sample im Batch
            - 'original_sample_ids': Array von originalen Sample-IDs (vor Filterung)
        """
        return extract_batch_metadata(dataloader, batch_idx=batch_idx)

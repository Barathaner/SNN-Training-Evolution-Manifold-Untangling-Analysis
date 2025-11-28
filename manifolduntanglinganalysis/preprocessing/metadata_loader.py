"""
Lädt Metadaten aus der CSV-Datei für gefilterte Datasets.
Dies stellt sicher, dass Trial-Nummern und Filenames konsistent bleiben.
"""
import pandas as pd
from typing import List, Optional, Dict
import numpy as np


class MetadataLoader:
    """
    Lädt und verwaltet Metadaten aus der CSV-Datei.
    Ermöglicht konsistente Metadaten auch nach Filterung des Datasets.
    """
    
    def __init__(self, csv_path: str = "shd_metadata_train.csv"):
        """
        Args:
            csv_path: Pfad zur CSV-Datei mit den Metadaten
        """
        self.df = pd.read_csv(csv_path)
        print(f"✅ Metadaten geladen: {len(self.df)} Samples aus {csv_path}")
    
    def get_metadata(self, sample_idx: int) -> Dict:
        """
        Gibt die Metadaten für einen bestimmten Sample-Index zurück.
        
        Args:
            sample_idx: Index im Original-Dataset
            
        Returns:
            Dictionary mit allen Metadaten (inkl. Trial-Nummer und Filename)
        """
        row = self.df[self.df['sample_idx'] == sample_idx]
        if len(row) == 0:
            raise ValueError(f"Sample-Index {sample_idx} nicht in Metadaten gefunden!")
        return row.iloc[0].to_dict()
    
    def get_metadata_batch(self, sample_indices: List[int]) -> pd.DataFrame:
        """
        Gibt die Metadaten für mehrere Sample-Indizes zurück.
        
        Args:
            sample_indices: Liste von Indizes im Original-Dataset
            
        Returns:
            DataFrame mit Metadaten für alle angegebenen Indizes
        """
        return self.df[self.df['sample_idx'].isin(sample_indices)].copy()
    
    def get_trial_numbers(self, sample_indices: List[int]) -> np.ndarray:
        """
        Gibt die Trial-Nummern für mehrere Sample-Indizes zurück.
        
        Args:
            sample_indices: Liste von Indizes im Original-Dataset
            
        Returns:
            Array mit Trial-Nummern (in der gleichen Reihenfolge wie sample_indices)
        """
        metadata = self.get_metadata_batch(sample_indices)
        # Sortiere nach sample_idx, um die richtige Reihenfolge zu garantieren
        metadata = metadata.set_index('sample_idx')
        return metadata.loc[sample_indices, 'trial'].values
    
    def get_filenames(self, sample_indices: List[int]) -> List[str]:
        """
        Gibt die Original-Filenames für mehrere Sample-Indizes zurück.
        
        Args:
            sample_indices: Liste von Indizes im Original-Dataset
            
        Returns:
            Liste von Filenames (in der gleichen Reihenfolge wie sample_indices)
        """
        metadata = self.get_metadata_batch(sample_indices)
        metadata = metadata.set_index('sample_idx')
        return metadata.loc[sample_indices, 'original_filename'].tolist()
    
    def filter_and_get_metadata(self, **criteria) -> pd.DataFrame:
        """
        Filtert die Metadaten nach Kriterien und gibt die gefilterten Daten zurück.
        
        Args:
            **criteria: Filter-Kriterien (z.B. speaker=2, language='english', digit=5)
            
        Returns:
            DataFrame mit gefilterten Metadaten
            
        Examples:
            >>> loader = MetadataLoader()
            >>> # Finde alle Samples von Speaker 2, Digit 5
            >>> filtered = loader.filter_and_get_metadata(speaker=2, digit=5)
            >>> print(f"Gefunden: {len(filtered)} Samples")
            >>> sample_indices = filtered['sample_idx'].tolist()
        """
        filtered = self.df.copy()
        
        for key, value in criteria.items():
            if key not in filtered.columns:
                raise ValueError(f"Unbekanntes Kriterium: {key}. "
                               f"Verfügbare Spalten: {list(filtered.columns)}")
            filtered = filtered[filtered[key] == value]
        
        return filtered


def create_filtered_dataset_with_metadata(
    dataset,
    metadata_loader: MetadataLoader,
    filter_indices: List[int]
) -> tuple:
    """
    Erstellt ein gefiltertes Dataset und holt die entsprechenden Metadaten.
    
    Args:
        dataset: Das Original-Dataset (z.B. tonic.datasets.SHD)
        metadata_loader: MetadataLoader-Instanz
        filter_indices: Liste von Indizes, die behalten werden sollen
        
    Returns:
        Tuple von (filtered_samples, metadata_df)
        - filtered_samples: Liste der gefilterten Samples
        - metadata_df: DataFrame mit Metadaten für die gefilterten Samples
        
    Example:
        >>> import tonic
        >>> from manifolddatageneration.preprocessing.metadata_loader import (
        ...     MetadataLoader, create_filtered_dataset_with_metadata
        ... )
        >>> 
        >>> # Lade Dataset und Metadaten
        >>> dataset = tonic.datasets.SHD(save_to="./data", train=True)
        >>> loader = MetadataLoader("shd_metadata_train.csv")
        >>> 
        >>> # Filtere nach Speaker 2
        >>> filtered_meta = loader.filter_and_get_metadata(speaker=2)
        >>> filter_indices = filtered_meta['sample_idx'].tolist()
        >>> 
        >>> # Erstelle gefiltertes Dataset mit Metadaten
        >>> filtered_data, metadata = create_filtered_dataset_with_metadata(
        ...     dataset, loader, filter_indices
        ... )
        >>> 
        >>> # Jetzt haben wir:
        >>> # - filtered_data: Die Event-Daten
        >>> # - metadata: DataFrame mit Trial-Nummern, Filenames, etc.
        >>> print(f"Gefilterte Samples: {len(filtered_data)}")
        >>> print(f"Trial-Nummern: {metadata['trial'].tolist()}")
        >>> print(f"Filenames: {metadata['original_filename'].tolist()}")
    """
    # Hole gefilterte Samples aus dem Dataset
    filtered_samples = []
    for idx in filter_indices:
        events, label = dataset[idx]
        filtered_samples.append((events, label))
    
    # Hole Metadaten für die gefilterten Samples
    metadata_df = metadata_loader.get_metadata_batch(filter_indices)
    
    print(f"✅ Gefiltertes Dataset erstellt:")
    print(f"   - {len(filtered_samples)} Samples")
    print(f"   - Trial-Bereich: {metadata_df['trial'].min()}-{metadata_df['trial'].max()}")
    print(f"   - Labels: {sorted(metadata_df['label'].unique())}")
    
    return filtered_samples, metadata_df


if __name__ == "__main__":
    # Beispiel-Nutzung
    print("=" * 80)
    print("METADATA LOADER - BEISPIEL")
    print("=" * 80)
    
    # Lade Metadaten
    loader = MetadataLoader("shd_metadata_train.csv")
    
    # Beispiel 1: Metadaten für einen einzelnen Sample
    print("\n1️⃣  Metadaten für Sample 100:")
    metadata = loader.get_metadata(100)
    print(f"   Label: {metadata['label']}")
    print(f"   Word: {metadata['word']}")
    print(f"   Trial: {metadata['trial']}")
    print(f"   Filename: {metadata['original_filename']}")
    
    # Beispiel 2: Filtere nach Speaker 2, English
    print("\n2️⃣  Filtere nach Speaker 2, English:")
    filtered = loader.filter_and_get_metadata(speaker=2, language='english')
    print(f"   Gefunden: {len(filtered)} Samples")
    print(f"   Trial-Bereich: {filtered['trial'].min()}-{filtered['trial'].max()}")
    print(f"   Sample-Indizes (erste 10): {filtered['sample_idx'].tolist()[:10]}")
    
    # Beispiel 3: Hole Trial-Nummern für spezifische Indizes
    print("\n3️⃣  Trial-Nummern für Indizes [100, 200, 300]:")
    trials = loader.get_trial_numbers([100, 200, 300])
    print(f"   Trials: {trials}")
    
    print("\n" + "=" * 80)


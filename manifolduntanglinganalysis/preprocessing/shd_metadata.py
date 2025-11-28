"""
SHD Dataset Metadata Utilities
Funktionen zum Extrahieren von Metadaten aus dem SHD-Dataset.
"""
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional


class SHDMetadata:
    """
    Hilfsklasse zum Zugriff auf SHD-Metadaten.
    """
    
    def __init__(self, data_path: str = "./data/SHD", train: bool = True):
        """
        Initialisiert den Metadata-Reader.
        
        Args:
            data_path: Pfad zum SHD-Datensatz
            train: True für Trainingsdaten, False für Testdaten
        """
        self.data_path = Path(data_path)
        self.train = train
        self.h5_file = self.data_path / ("shd_train.h5" if train else "shd_test.h5")
        
        if not self.h5_file.exists():
            raise FileNotFoundError(f"H5-Datei nicht gefunden: {self.h5_file}")
        
        # Lade Metadaten
        with h5py.File(self.h5_file, 'r') as f:
            self.labels = f['labels'][:]
            self.speakers = f['extra']['speaker'][:]
            self.keys = [k.decode('utf-8') for k in f['extra']['keys'][:]]
            
            # Speaker Meta-Info
            self.speaker_genders = [g.decode('utf-8') for g in f['extra']['meta_info']['gender'][:]]
            self.speaker_ages = f['extra']['meta_info']['age'][:]
            self.speaker_heights = f['extra']['meta_info']['body_height'][:]
        
        # Berechne Trial-Nummern für alle Samples
        self._compute_trials()
    
    def _compute_trials(self):
        """
        Berechnet die Trial-Nummern für jedes Sample.
        
        WICHTIG: Die absoluten Trial-Nummern aus den Original-Dateinamen 
        sind NICHT in der H5-Datei gespeichert und können nicht exakt 
        rekonstruiert werden.
        
        Diese Funktion berechnet RELATIVE Trial-Nummern (beginnend bei 0).
        Trials werden SEPARAT für jede (Speaker, Language) Kombination gezählt!
        
        Ein Trial ist eine Sequenz von aufsteigenden Digits (z.B. 0-9 oder 0-5).
        Ein neues Trial beginnt, wenn der Digit-Wert kleiner oder gleich dem 
        höchsten Digit im aktuellen Trial ist.
        """
        self.trials = np.zeros(len(self.labels), dtype=int)
        
        # Gruppiere Samples nach (Speaker, Language) und sortiere nach Index
        speaker_lang_samples = defaultdict(list)
        for idx in range(len(self.labels)):
            speaker = self.speakers[idx]
            label = self.labels[idx]
            language, digit = self._label_to_language_and_digit(label)
            # Verwende (speaker, language) als Key
            speaker_lang_samples[(speaker, language)].append((idx, digit))
        
        # Für jede (Speaker, Language) Kombination: Berechne Trial-Nummern
        for (speaker, language), samples in speaker_lang_samples.items():
            current_trial = 0
            max_digit_in_trial = -1  # Höchster Digit im aktuellen Trial
            
            for idx, digit in samples:
                # Neues Trial beginnt wenn:
                # - Wir zu Digit 0 zurückkehren nach ≥7 ODER
                # - Wir zu Digit 1 zurückkehren nach ≥9 (nur nach vollständiger Sequenz)
                # 
                # Dies ist die beste Balance zwischen:
                # - Max Trials ≤ 64
                # - Durchschnitt 35-45 Trials pro (Speaker, Language)
                # - Max ~10 Samples pro Trial
                # 
                # Beispiele:
                # - 9 → 0: Neues Trial ✓
                # - 8 → 0: Neues Trial ✓
                # - 7 → 0: Neues Trial ✓
                # - 9 → 1: Neues Trial ✓
                # - 8 → 1: Kein neues Trial
                # - 6 → 0: Kein neues Trial
                
                if (max_digit_in_trial >= 7 and digit == 0) or (max_digit_in_trial >= 9 and digit == 1):
                    # Sequenz-Reset erkannt → neues Trial
                    current_trial += 1
                    max_digit_in_trial = digit
                else:
                    # Digit steigt weiter an oder ist kein vollständiger Reset
                    max_digit_in_trial = max(max_digit_in_trial, digit)
                
                self.trials[idx] = current_trial
    
    def find_samples(self, return_metadata: Optional[str] = None, **criteria):
        """
        Findet alle Samples, die bestimmte Kriterien erfüllen.
        
        Unterstützte Kriterien:
            - label: Word ID (0-19)
            - word: Wortname (z.B. 'zero', 'eins')
            - language: 'english' oder 'german'
            - digit: Ziffer 0-9
            - speaker: Speaker ID
            - trial: Trial-Nummer
            - gender: 'male' oder 'female' (bezieht sich auf Speaker)
            - min_age, max_age: Altersfilter (Speaker)
            - min_height, max_height: Größenfilter (Speaker, in cm)
        
        Args:
            return_metadata: Bestimmt das Rückgabeformat:
                - None: Nur Liste von Indices (default)
                - 'label': Liste von Dicts mit {idx, label}
                - 'basic': Liste von Dicts mit {idx, label, word, speaker}
                - 'full': Liste von Dicts mit allen Metadaten
            **criteria: Beliebige Kombination der obigen Kriterien
        
        Returns:
            Liste von Sample-Indices (wenn return_metadata=None)
            oder Liste von Dicts (wenn return_metadata angegeben)
        
        Examples:
            >>> # Nur Indices
            >>> metadata.find_samples(gender='male')
            [0, 1, 4, ...]
            
            >>> # Mit Labels
            >>> metadata.find_samples(return_metadata='label', gender='male')
            [{'idx': 0, 'label': 11}, {'idx': 1, 'label': 13}, ...]
            
            >>> # Mit Basic-Info
            >>> metadata.find_samples(return_metadata='basic', word='zero')
            [{'idx': 9, 'label': 0, 'word': 'zero', 'speaker': 3}, ...]
            
            >>> # Mit vollständigen Metadaten
            >>> metadata.find_samples(return_metadata='full', speaker=3)
            [{'idx': 9, 'label': 0, 'word': 'zero', ...}, ...]
        """
        matching = []
        
        for idx in range(len(self.labels)):
            label = self.labels[idx]
            speaker = self.speakers[idx]
            trial = self.trials[idx]
            language, digit = self._label_to_language_and_digit(label)
            word = self.keys[label]
            
            # Prüfe alle Kriterien
            match = True
            
            # Label
            if 'label' in criteria and criteria['label'] != label:
                match = False
            
            # Word
            if 'word' in criteria and criteria['word'] != word:
                match = False
            
            # Language
            if 'language' in criteria and criteria['language'] != language:
                match = False
            
            # Digit
            if 'digit' in criteria and criteria['digit'] != digit:
                match = False
            
            # Speaker
            if 'speaker' in criteria and criteria['speaker'] != speaker:
                match = False
            
            # Trial
            if 'trial' in criteria and criteria['trial'] != trial:
                match = False
            
            # Gender (von Speaker)
            if 'gender' in criteria:
                speaker_gender = self.speaker_genders[speaker]
                if criteria['gender'] != speaker_gender:
                    match = False
            
            # Age filters
            if 'min_age' in criteria:
                speaker_age = self.speaker_ages[speaker]
                if speaker_age < criteria['min_age']:
                    match = False
            
            if 'max_age' in criteria:
                speaker_age = self.speaker_ages[speaker]
                if speaker_age > criteria['max_age']:
                    match = False
            
            # Height filters
            if 'min_height' in criteria:
                speaker_height = self.speaker_heights[speaker]
                if speaker_height < criteria['min_height']:
                    match = False
            
            if 'max_height' in criteria:
                speaker_height = self.speaker_heights[speaker]
                if speaker_height > criteria['max_height']:
                    match = False
            
            if match:
                # Füge basierend auf return_metadata hinzu
                if return_metadata is None:
                    # Nur Index
                    matching.append(idx)
                elif return_metadata == 'label':
                    # Index + Label
                    matching.append({'idx': idx, 'label': label})
                elif return_metadata == 'basic':
                    # Index + Label + Word + Speaker
                    matching.append({
                        'idx': idx,
                        'label': label,
                        'word': word,
                        'speaker': speaker
                    })
                elif return_metadata == 'full':
                    # Vollständige Metadaten
                    matching.append(self.get_metadata(idx))
                else:
                    raise ValueError(f"Unbekannter return_metadata Wert: {return_metadata}. "
                                   f"Erlaubt: None, 'label', 'basic', 'full'")
        
        return matching
    
    def _label_to_language_and_digit(self, label: int) -> Tuple[str, int]:
        """
        Konvertiert Label zu Sprache und Ziffer.
        
        Args:
            label: Word ID (0-19)
        
        Returns:
            (language, digit) Tuple
            - language: "english" oder "german"
            - digit: Ziffer 0-9
        """
        if label < 10:
            # English: 0-9
            return "english", label
        elif label == 10:
            # null = 0 auf Deutsch
            return "german", 0
        else:
            # German: 11-19 → digits 1-9
            return "german", label - 10
    
    def get_original_filename(self, sample_idx: int) -> str:
        """
        Rekonstruiert den Original-Dateinamen für einen Sample-Index.
        
        Format: lang-{language}_speaker-{speaker_id:02d}_trial-{trial}_digit-{digit}.flac
        
        Args:
            sample_idx: Index im Dataset (0-basiert)
        
        Returns:
            Original-Dateiname als String
        
        Example:
            >>> metadata = SHDMetadata()
            >>> metadata.get_original_filename(9)
            'lang-english_speaker-03_trial-0_digit-0.flac'
        """
        if sample_idx < 0 or sample_idx >= len(self.labels):
            raise IndexError(f"Sample-Index {sample_idx} außerhalb des gültigen Bereichs (0-{len(self.labels)-1})")
        
        # Hole Metadaten
        label = self.labels[sample_idx]
        speaker = self.speakers[sample_idx]
        trial = self.trials[sample_idx]
        
        # Konvertiere Label zu Language und Digit
        language, digit = self._label_to_language_and_digit(label)
        
        # Konstruiere Filename
        filename = f"lang-{language}_speaker-{speaker:02d}_trial-{trial}_digit-{digit}.flac"
        
        return filename
    
    def get_metadata(self, sample_idx: int) -> Dict:
        """
        Gibt alle Metadaten für ein Sample zurück.
        
        Args:
            sample_idx: Index im Dataset
        
        Returns:
            Dictionary mit allen Metadaten
        """
        if sample_idx < 0 or sample_idx >= len(self.labels):
            raise IndexError(f"Sample-Index {sample_idx} außerhalb des gültigen Bereichs")
        
        label = self.labels[sample_idx]
        speaker = self.speakers[sample_idx]
        trial = self.trials[sample_idx]
        language, digit = self._label_to_language_and_digit(label)
        
        return {
            'sample_idx': sample_idx,
            'label': label,
            'word': self.keys[label],
            'language': language,
            'digit': digit,
            'speaker': speaker,
            'trial': trial,
            'original_filename': self.get_original_filename(sample_idx),
            'speaker_gender': self.speaker_genders[speaker],
            'speaker_age': self.speaker_ages[speaker],
            'speaker_height': self.speaker_heights[speaker]
        }
    
    def __len__(self):
        return len(self.labels)
    
    def __repr__(self):
        return f"SHDMetadata(train={self.train}, samples={len(self)})"


# Convenience-Funktion
def get_original_filename(sample_idx: int, train: bool = True, data_path: str = "./data/SHD") -> str:
    """
    Convenience-Funktion zum schnellen Abrufen des Original-Dateinamens.
    
    Args:
        sample_idx: Index im Dataset
        train: True für Training, False für Test
        data_path: Pfad zum Dataset
    
    Returns:
        Original-Dateiname als String
    
    Example:
        >>> from manifolddatageneration.preprocessing.shd_metadata import get_original_filename
        >>> filename = get_original_filename(9, train=True)
        >>> print(filename)
        'lang-english_speaker-03_trial-0_digit-0.flac'
    """
    metadata = SHDMetadata(data_path=data_path, train=train)
    return metadata.get_original_filename(sample_idx)


if __name__ == "__main__":
    # Test der Funktionalität
    print("=" * 70)
    print("TEST: Original-Dateinamen-Rekonstruktion")
    print("=" * 70)
    
    metadata = SHDMetadata(train=True)
    
    print(f"\nDataset: {metadata}")
    
    # Teste für einige Beispiel-Indices
    test_indices = [0, 1, 2, 9, 17, 100, 500]
    
    print(f"\n{'Index':>5} | {'Label':>5} | {'Word':>10} | {'Speaker':>7} | {'Trial':>5} | Original Filename")
    print("-" * 100)
    
    for idx in test_indices:
        meta = metadata.get_metadata(idx)
        print(f"{idx:5d} | {meta['label']:5d} | {meta['word']:>10s} | "
              f"{meta['speaker']:7d} | {meta['trial']:5d} | {meta['original_filename']}")
    
    # Zeige auch Speaker-Info für ein Beispiel
    print("\n" + "=" * 70)
    print("DETAILLIERTE METADATEN FÜR SAMPLE 9")
    print("=" * 70)
    
    meta = metadata.get_metadata(9)
    for key, value in meta.items():
        print(f"  {key:20s}: {value}")


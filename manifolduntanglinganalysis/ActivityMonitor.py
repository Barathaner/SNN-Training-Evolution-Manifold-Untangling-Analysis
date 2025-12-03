import torch
from collections import defaultdict
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Callable, Union
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

# Import MetadataExtractor von metadata_extractor um Duplikate zu vermeiden
try:
    from manifolduntanglinganalysis.preprocessing.metadata_extractor import MetadataExtractor
except ImportError:
    # Fallback falls Import nicht möglich
    class MetadataExtractor(ABC):
        """
        Abstrakte Basisklasse für Metadaten-Extraktoren.
        Ermöglicht verschiedene Implementierungen für verschiedene Datasets.
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


class ActivityMonitor:
    def __init__(self, 
                 model,
                 metadata_extractor: Optional[MetadataExtractor] = None,
                 input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Initialisiert den ActivityMonitor mit leeren Buffern für Spike-Aktivitäten.
        
        Args:
            model: Das PyTorch-Modell, das überwacht werden soll
            metadata_extractor: Optional, Extractor für Metadaten (für verschiedene Datasets)
            input_transform: Optional, Funktion zur Transformation der Input-Daten
                           (z.B. lambda x: x.squeeze(2) if x.ndim == 4 else x)
        """
        self.model = model
        self.metadata_extractor = metadata_extractor
        self.input_transform = input_transform
        # Dictionary: layer_name -> Liste von Spike-Tensoren (ein Tensor pro Zeitschritt)
        self.spike_buffer = defaultdict(list)
        # Dictionary: layer_name -> Liste von Input-Tensoren (optional, für Debugging)
        self.input_buffer = defaultdict(list)
        # Speichert die Layer-Namen für jeden registrierten Hook
        self.layer_names = {}
        # Hooks am gesamten Modell (werden beim enable_monitoring registriert)
        self.model_hooks = []
        # Hooks an einzelnen Layern
        self.layer_hooks = []
        # Flag, um zu wissen, ob ein Forward-Pass gerade läuft
        self.forward_pass_active = False
        # Aktuelles Label für den laufenden Forward-Pass
        self.current_labels = None
        # Metadaten pro Sample im aktuellen Batch
        # Format: {'sample_0': {'speaker': 5, 'original_sample_id': 123, ...}, ...}
        self.batch_metadata = {}
    
    def _model_pre_forward_hook(self, module, input):
        """
        Wird VOR dem Forward-Pass des GESAMTEN Modells aufgerufen.
        Wird genau EINMAL pro Forward-Pass aufgerufen (nicht pro Zeitschritt!).
        Hier löschen wir den Buffer, um für einen neuen Forward-Pass bereit zu sein.
        
        Das aktuelle Label kann über self.current_labels abgerufen werden,
        wenn es vorher mit set_current_labels() gesetzt wurde.
        """
        self.clear_buffer()
        self.forward_pass_active = True
        # self.current_labels ist hier verfügbar, wenn es vorher gesetzt wurde
    
    def _model_forward_hook(self, module, input, output):
        """
        Wird NACH dem Forward-Pass des GESAMTEN Modells aufgerufen.
        Verwendet register_forward_hook (wird nach dem Forward-Pass aufgerufen).
        Wird genau EINMAL pro Forward-Pass aufgerufen, NACH allen Zeitschritten.
        Hier wissen wir, dass alle Zeitschritte durchgelaufen sind.
        """
        self.forward_pass_active = False
    
    def pre_forward_hook(self, module, input):
        """
        Wird VOR jedem Forward-Pass eines einzelnen Moduls aufgerufen.
        Wird bei jedem Zeitschritt für jeden Layer aufgerufen.
        """
        # Optional: Input für Debugging speichern
        # layer_name = self.layer_names.get(id(module), module.__class__.__name__)
        # self.input_buffer[layer_name].append(input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu())
        pass
    
    def forward_hook(self, module, input, output):
        """
        Wird NACH jedem Forward-Pass eines Moduls aufgerufen.
        Hier sammeln wir die Spikes pro Layer über alle Zeitschritte.
        
        Args:
            module: Das Modul, das den Hook ausgelöst hat
            input: Input zum Modul
            output: Output vom Modul (bei LIF-Layern: (spk, mem) Tupel)
        """
        # Hole den Layer-Namen (wird beim Registrieren gesetzt)
        layer_name = self.layer_names.get(id(module), module.__class__.__name__)
        
        # Bei snntorch LIF-Layern ist output ein Tupel (spk, mem)
        if isinstance(output, tuple) and len(output) >= 1:
            spikes = output[0]  # Erste Komponente sind die Spikes
        else:
            # Falls kein Tupel, nehmen wir den Output direkt
            spikes = output
        
        # Spikes detachen und auf CPU kopieren (wichtig für Speicher!)
        # Shape: [Batch, Features] pro Zeitschritt
        spikes_detached = spikes.detach().cpu()
        
        # Füge die Spikes für diesen Zeitschritt zur Liste hinzu
        self.spike_buffer[layer_name].append(spikes_detached)
    
    def full_backward_hook(self, module, grad_input, grad_output):
        """
        Wird während des Backward-Passes aufgerufen.
        """
        # Optional: Gradienten für Debugging speichern
        pass
    
    def enable_monitoring(self, lif_layer_names=None, verbose=True):
        """
        Aktiviert das Monitoring für das gesamte Modell.
        Registriert Hooks am Modell und an allen Layern, um Spikes zu sammeln.
        
        Args:
            lif_layer_names: Optional, Liste von Layer-Namen, die überwacht werden sollen.
                            Wenn None, werden alle Leaf-Module überwacht.
            verbose: Wenn True, werden registrierte Layer ausgegeben
        """
        # 1. Hooks am gesamten Modell (um zu erkennen, wann Forward-Pass beginnt/endet)
        pre_hook = self.model.register_forward_pre_hook(self._model_pre_forward_hook)
        forward_hook = self.model.register_forward_hook(self._model_forward_hook)
        self.model_hooks.extend([pre_hook, forward_hook])
        
        # 2. Hooks an einzelnen Layern (um Spikes zu sammeln)
        if lif_layer_names is None:
            # Wenn keine Layer-Namen angegeben, überwache alle Leaf-Module
            lif_layer_names = []
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Nur Leaf-Module (keine Container)
                # Registriere Layer-Namen
                self.layer_names[id(module)] = name
                
                # Registriere Hooks für diesen Layer
                pre_hook = module.register_forward_pre_hook(self.pre_forward_hook)
                forward_hook = module.register_forward_hook(self.forward_hook)
                backward_hook = module.register_full_backward_hook(self.full_backward_hook)
                self.layer_hooks.extend([pre_hook, forward_hook, backward_hook])
                
                if verbose:
                    if name in lif_layer_names:
                        print(f"✅ Registered hook for LIF-Layer: {name}")
                    else:
                        print(f"Registered hook for {name}")
            else:
                if verbose:
                    print(f"Skipping {name} because it has children")
    
    def disable_monitoring(self):
        """
        Deaktiviert das Monitoring und entfernt alle Hooks.
        """
        # Entferne Hooks am Modell
        for hook in self.model_hooks:
            hook.remove()
        self.model_hooks.clear()
        
        # Entferne Hooks an einzelnen Layern
        for hook in self.layer_hooks:
            hook.remove()
        self.layer_hooks.clear()
        
        self.forward_pass_active = False
    
    def set_current_labels(self, labels):
        """
        Setzt die Labels für den nächsten Forward-Pass.
        Diese Labels sind dann im _model_pre_forward_hook über self.current_labels verfügbar.
        
        Args:
            labels: Tensor oder Liste von Labels für den aktuellen Batch
        """
        if isinstance(labels, torch.Tensor):
            self.current_labels = labels.detach().cpu()
        else:
            self.current_labels = labels
    
    def get_current_labels(self):
        """
        Gibt die Labels des aktuellen Forward-Passes zurück.
        
        Returns:
            Labels des aktuellen Forward-Passes oder None
        """
        return self.current_labels
    
    def set_batch_metadata(self, metadata: Dict[str, np.ndarray]):
        """
        Setzt Metadaten für den aktuellen Batch.
        Die Metadaten werden pro Sample im Batch gespeichert.
        
        Args:
            metadata: Dictionary mit Arrays für jeden Metadaten-Typ.
                     Keys sollten z.B. 'speakers', 'original_sample_ids' sein.
                     Die Arrays müssen die gleiche Länge haben (Batch-Größe).
        
        Beispiel:
            metadata = {
                'speakers': np.array([5, 3, 7, ...]),
                'original_sample_ids': np.array([123, 456, 789, ...])
            }
            monitor.set_batch_metadata(metadata)
        """
        # Konvertiere zu Dictionary pro Sample
        self.batch_metadata = {}
        
        # Bestimme Batch-Größe aus den ersten Array
        if not metadata:
            return
        
        first_key = list(metadata.keys())[0]
        batch_size = len(metadata[first_key])
        
        # Erstelle Dictionary für jeden Sample im Batch
        for i in range(batch_size):
            sample_metadata = {}
            for key, values in metadata.items():
                if isinstance(values, (np.ndarray, list, torch.Tensor)):
                    if isinstance(values, torch.Tensor):
                        sample_metadata[key] = values[i].item() if values[i].numel() == 1 else values[i].cpu().numpy()
                    else:
                        sample_metadata[key] = values[i].item() if np.isscalar(values[i]) else values[i]
                else:
                    sample_metadata[key] = values
            self.batch_metadata[f'sample_{i}'] = sample_metadata
    
    def get_batch_metadata(self) -> Dict:
        """
        Gibt die Metadaten des aktuellen Batches zurück.
        
        Returns:
            Dictionary mit Metadaten pro Sample: {'sample_0': {...}, 'sample_1': {...}, ...}
        """
        return self.batch_metadata
    
    def clear_buffer(self):
        """
        Löscht alle gesammelten Spike-Daten.
        Sollte vor jedem neuen Forward-Pass aufgerufen werden.
        """
        self.spike_buffer.clear()
        self.input_buffer.clear()
        # Labels werden nicht automatisch gelöscht, damit sie nach dem Forward-Pass verfügbar sind
        # Metadaten werden auch nicht automatisch gelöscht
    
    def get_spikes(self, layer_name=None):
        """
        Gibt die gesammelten Spikes zurück.
        
        Args:
            layer_name: Optional, spezifischer Layer-Name. Wenn None, werden alle Layer zurückgegeben.
        
        Returns:
            Dictionary: {layer_name: Tensor} mit Shape [T, Batch, Features]
                        oder einzelner Tensor wenn layer_name angegeben
        """
        result = {}
        
        for name, spike_list in self.spike_buffer.items():
            if layer_name is None or name == layer_name:
                if spike_list:
                    # Stack über Zeitschritte: [T, Batch, Features]
                    # spike_list ist eine Liste von [Batch, Features] Tensoren
                    stacked = torch.stack(spike_list, dim=0)  # [T, Batch, Features]
                    result[name] = stacked
        
        if layer_name is not None:
            return result.get(layer_name, None)
        return result
    
    def get_spikes_bt(self, layer_name=None):
        """
        Gibt die gesammelten Spikes zurück, transponiert zu [Batch, T, Features].
        
        Args:
            layer_name: Optional, spezifischer Layer-Name. Wenn None, werden alle Layer zurückgegeben.
        
        Returns:
            Dictionary: {layer_name: Tensor} mit Shape [Batch, T, Features]
                        oder einzelner Tensor wenn layer_name angegeben
        """
        spikes_tb = self.get_spikes(layer_name)
        
        if isinstance(spikes_tb, dict):
            return {name: tensor.permute(1, 0, 2) for name, tensor in spikes_tb.items()}
        elif spikes_tb is not None:
            return spikes_tb.permute(1, 0, 2)
        return spikes_tb
    
    def _convert_spikes_to_tonic_format(self, spikes: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Konvertiert Spikes von [Batch, T, Features] in Tonic-Format (t, x, p) pro Sample.
        
        Args:
            spikes: Spikes als numpy Array [Batch, T, Features]
        
        Returns:
            Dictionary: {sample_idx: events_array} wobei events_array Shape [N_events, 3] hat
                       mit Spalten (t, x, p)
        """
        batch_size, T, num_features = spikes.shape
        events_per_sample = {}
        
        for sample_idx in range(batch_size):
            # Hole Spikes für dieses Sample: [T, Features]
            sample_spikes = spikes[sample_idx]
            
            # Finde alle Spikes (wo spike == 1)
            spike_indices = np.where(sample_spikes == 1)
            
            # Erstelle Events im Format (t, x, p)
            # t = Zeitschritt, x = Feature-Index, p = Polarity (immer 1)
            events = np.column_stack([
                spike_indices[0],  # t (Zeitschritt)
                spike_indices[1],  # x (Feature/Neuron-Index)
                np.ones(len(spike_indices[0]), dtype=np.uint8)  # p (Polarity, immer 1)
            ])
            
            # Konvertiere zu strukturiertem Array (wie in Tonic)
            dtype = np.dtype([('t', 'uint64'), ('x', 'uint16'), ('p', 'uint8')])
            events_structured = np.empty(len(events), dtype=dtype)
            events_structured['t'] = events[:, 0].astype(np.uint64)
            events_structured['x'] = events[:, 1].astype(np.uint16)
            events_structured['p'] = events[:, 2].astype(np.uint8)
            
            events_per_sample[sample_idx] = events_structured
        
        return events_per_sample
    
    def save_as_h5(self, 
                   layer_name: str,
                   save_dir: str,
                   filename: Optional[str] = None,
                   epoch: Optional[int] = None,
                   include_metadata: bool = True,
                   spikes_data: Optional[Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray]]] = None,
                   metadata: Optional[Dict] = None,
                   labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> str:
        """
        Speichert die Spikes eines Layers als HDF5-Datei im Tonic-Format (t, x, p).
        
        Das Format entspricht dem Spiking Heidelberg Digits Datensatz:
        - t: Zeit (Zeitschritt)
        - x: Neuron/Feature-Index
        - p: Polarity (hier immer 1, da binäre Spikes)
        
        Args:
            layer_name: Name des Layers (z.B. 'lif1')
            save_dir: Verzeichnis zum Speichern (z.B. 'data/activity_logs')
            filename: Optional, custom Dateiname. Wenn None, wird automatisch generiert.
            epoch: Optional, Epoch-Nummer für automatischen Dateinamen
            include_metadata: Wenn True, werden Metadaten und Labels mitgespeichert
            spikes_data: Optional, direkt übergebene Spike-Daten [Batch, T, Features]
                       (numpy array, torch Tensor oder Dict). Wenn None, wird aus Buffer gelesen.
            metadata: Optional, Metadaten-Dictionary (überschreibt self.batch_metadata)
            labels: Optional, Labels-Array (überschreibt self.current_labels)
        
        Returns:
            Pfad zur gespeicherten Datei
        
        Beispiel:
            # Automatischer Dateiname: epoch_001_lif1_spk_events.h5
            path = monitor.save_as_h5('lif1', 'data/activity_logs', epoch=1)
            
            # Mit direkt übergebenen Daten
            path = monitor.save_as_h5('lif1', 'data/activity_logs', 
                                     spikes_data=spikes_array, 
                                     metadata=my_metadata,
                                     labels=my_labels)
        """
        # Hole Spikes: Entweder direkt übergeben oder aus Buffer
        if spikes_data is not None:
            if isinstance(spikes_data, dict):
                spikes = spikes_data.get(layer_name)
            elif isinstance(spikes_data, (np.ndarray, torch.Tensor)):
                spikes = spikes_data
            else:
                raise ValueError(f"Ungültiger Typ für spikes_data: {type(spikes_data)}")
        else:
            # Fallback: Aus Buffer lesen
            spikes = self.get_spikes_bt(layer_name=layer_name)
        
        if spikes is None:
            raise ValueError(f"Keine Spikes für Layer '{layer_name}' gefunden")
        
        # Konvertiere zu numpy
        if isinstance(spikes, torch.Tensor):
            spikes_np = spikes.numpy()
        else:
            spikes_np = np.asarray(spikes)
        
        # Validiere Shape
        if spikes_np.ndim != 3:
            raise ValueError(f"Spikes müssen Shape [Batch, T, Features] haben, bekam {spikes_np.shape}")
        
        # Konvertiere zu Tonic-Format (t, x, p) pro Sample
        events_per_sample = self._convert_spikes_to_tonic_format(spikes_np)
        
        # Erstelle Verzeichnis
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generiere Dateinamen
        if filename is None:
            epoch_str = f"epoch_{epoch:03d}_" if epoch is not None else ""
            filename = f"{epoch_str}{layer_name}_spk_events.h5"
        
        filepath = save_path / filename
        
        # Speichere als HDF5 im Tonic-Format
        with h5py.File(filepath, 'w') as f:
            # Speichere Events für jeden Sample im Batch
            events_group = f.create_group('events')
            
            for sample_idx, events in events_per_sample.items():
                # Erstelle Dataset für diesen Sample
                sample_key = f'sample_{sample_idx}'
                events_dataset = events_group.create_dataset(
                    sample_key,
                    data=events,
                    compression='gzip'
                )
                events_dataset.attrs['num_events'] = len(events)
                events_dataset.attrs['time_range'] = (events['t'].min(), events['t'].max()) if len(events) > 0 else (0, 0)
                events_dataset.attrs['num_features'] = spikes_np.shape[2]
            
            # Layer-Informationen
            f.attrs['layer_name'] = layer_name
            f.attrs['batch_size'] = spikes_np.shape[0]
            f.attrs['time_steps'] = spikes_np.shape[1]
            f.attrs['num_features'] = spikes_np.shape[2]
            f.attrs['format'] = 'tonic_events'  # (t, x, p)
            
            # Metadaten speichern, wenn vorhanden
            metadata_to_save = metadata if metadata is not None else self.batch_metadata
            if include_metadata and metadata_to_save:
                metadata_group = f.create_group('metadata')
                for sample_key, sample_meta in metadata_to_save.items():
                    sample_group = metadata_group.create_group(sample_key)
                    for key, value in sample_meta.items():
                        if isinstance(value, (int, float, np.number)):
                            sample_group.attrs[key] = value
                        elif isinstance(value, (np.ndarray, list)):
                            sample_group.create_dataset(key, data=np.array(value))
                        else:
                            sample_group.attrs[key] = str(value)
            
            # Labels speichern, wenn vorhanden
            labels_to_save = labels if labels is not None else self.current_labels
            if include_metadata and labels_to_save is not None:
                labels_np = labels_to_save.numpy() if isinstance(labels_to_save, torch.Tensor) else np.asarray(labels_to_save)
                f.create_dataset('labels', data=labels_np, compression='gzip')
                f['labels'].attrs['description'] = 'Ground truth labels for each sample in the batch'
        
        return str(filepath)
    
    def monitor_and_save_samples(self,
                                 dataloader: DataLoader,
                                 num_samples: int,
                                 layer_names: List[str],
                                 save_dir: str,
                                 epoch: Optional[int] = None,
                                 device: torch.device = torch.device('cpu'),
                                 verbose: bool = True,
                                 metadata_extractor: Optional[MetadataExtractor] = None,
                                 input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Dict[str, str]:
        """
        Überwacht und speichert Spikes für eine bestimmte Anzahl von Samples.
        Verwendet immer die gleichen Samples (die ersten N) für Vergleichbarkeit.
        Sammelt Spikes über alle Batches und speichert am Ende.
        
        Args:
            dataloader: DataLoader für die Daten
            num_samples: Anzahl der Samples, die überwacht werden sollen
            layer_names: Liste der Layer-Namen, die überwacht werden sollen
            save_dir: Verzeichnis zum Speichern der HDF5-Dateien
            epoch: Optional, Epoch-Nummer für Dateinamen
            device: Device für die Berechnung
            verbose: Wenn True, werden Fortschrittsmeldungen ausgegeben
            metadata_extractor: Optional, Extractor für Metadaten (überschreibt self.metadata_extractor)
            input_transform: Optional, Funktion zur Input-Transformation (überschreibt self.input_transform)
        
        Returns:
            Dictionary mit gespeicherten Dateipfaden: {layer_name: filepath}
        
        Beispiel:
            filepaths = monitor.monitor_and_save_samples(
                dataloader=test_dataloader,
                num_samples=64,
                layer_names=['lif0', 'lif1', 'lif2', 'lif3'],
                save_dir='data/activity_logs',
                epoch=1
            )
        """
        # Verwende übergebene Extractor/Transform oder die aus __init__
        extractor = metadata_extractor or self.metadata_extractor
        transform = input_transform or self.input_transform
        
        # Berechne, wie viele Batches benötigt werden
        batch_size = dataloader.batch_size
        num_batches_needed = (num_samples + batch_size - 1) // batch_size  # Aufrunden
        
        if verbose:
            print(f"📊 Überwache {num_samples} Samples ({num_batches_needed} Batches)...")
        
        # Akkumuliere Spikes und Metadaten über alle Batches
        all_spikes_accumulated = {layer: [] for layer in layer_names}
        all_metadata_accumulated = []
        all_labels_accumulated = []
        
        # Iteriere über Batches
        samples_collected = 0
        batch_idx = 0
        
        for events, labels in dataloader:
            if samples_collected >= num_samples:
                break
            
            # Berechne, wie viele Samples aus diesem Batch wir nehmen
            samples_in_batch = min(batch_size, num_samples - samples_collected)
            
            # Events vorbereiten
            events = events.to(device).float()
            # Wende Input-Transformation an, falls vorhanden
            if transform is not None:
                events = transform(events)
            # Fallback: Standard-Transformation für SHD-Format (wenn keine Transform angegeben)
            elif events.ndim == 4:
                events = events.squeeze(2)
            
            labels = labels.to(device)
            
            # Nimm nur die benötigten Samples aus dem Batch
            events = events[:samples_in_batch]
            labels = labels[:samples_in_batch]
            
            # Metadaten extrahieren (nur wenn Extractor vorhanden)
            batch_metadata_dict = {}
            if extractor is not None:
                try:
                    batch_metadata = extractor.extract(dataloader, batch_idx)
                    # Nimm nur die benötigten Samples aus den Metadaten
                    for key in batch_metadata:
                        batch_metadata[key] = batch_metadata[key][:samples_in_batch]
                    batch_metadata_dict = batch_metadata
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Warnung: Metadaten konnten nicht extrahiert werden: {e}")
            
            # Labels setzen
            self.set_current_labels(labels)
            
            # Forward-Pass (Spikes werden automatisch gesammelt)
            with torch.no_grad():
                _ = self.model(events)
            
            # Hole Spikes für diesen Batch und akkumuliere sie
            batch_spikes = self.get_spikes_bt()
            for layer_name in layer_names:
                if layer_name in batch_spikes:
                    # Konvertiere zu numpy und füge hinzu
                    spikes_np = batch_spikes[layer_name].numpy() if isinstance(batch_spikes[layer_name], torch.Tensor) else batch_spikes[layer_name]
                    all_spikes_accumulated[layer_name].append(spikes_np)
            
            # Akkumuliere Metadaten und Labels
            for i in range(samples_in_batch):
                sample_meta = {}
                for key, values in batch_metadata_dict.items():
                    sample_meta[key] = values[i] if isinstance(values, np.ndarray) else values
                all_metadata_accumulated.append(sample_meta)
            
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_labels_accumulated.append(labels_np)
            
            samples_collected += samples_in_batch
            batch_idx += 1
            
            if verbose:
                print(f"   ✅ {samples_collected}/{num_samples} Samples gesammelt")
        
        # Kombiniere alle gesammelten Spikes
        combined_spikes = {}
        for layer_name in layer_names:
            if all_spikes_accumulated[layer_name]:
                # Konkateniere über Batch-Dimension: [B1, T, F] + [B2, T, F] -> [B1+B2, T, F]
                combined_spikes[layer_name] = np.concatenate(all_spikes_accumulated[layer_name], axis=0)
        
        # Kombiniere Labels
        combined_labels = np.concatenate(all_labels_accumulated, axis=0) if all_labels_accumulated else None
        
        # Setze kombinierte Metadaten
        if all_metadata_accumulated:
            # Konvertiere zu Dictionary-Format für set_batch_metadata
            combined_metadata = {}
            for key in all_metadata_accumulated[0].keys():
                combined_metadata[key] = np.array([meta[key] for meta in all_metadata_accumulated])
            self.set_batch_metadata(combined_metadata)
        
        # Setze kombinierte Labels
        if combined_labels is not None:
            self.set_current_labels(torch.from_numpy(combined_labels))
        
        # Speichere Spikes für jeden Layer (mit allen Samples)
        saved_filepaths = {}
        # Bereite Metadaten für Speicherung vor
        combined_metadata_dict = {}
        if all_metadata_accumulated:
            for key in all_metadata_accumulated[0].keys():
                combined_metadata_dict[key] = np.array([meta[key] for meta in all_metadata_accumulated])
            # Konvertiere zu sample-basiertem Format
            metadata_for_save = {}
            for i in range(len(all_metadata_accumulated)):
                metadata_for_save[f'sample_{i}'] = all_metadata_accumulated[i]
        else:
            metadata_for_save = None
        
        for layer_name in layer_names:
            if layer_name in combined_spikes:
                try:
                    filepath = self.save_as_h5(
                        layer_name=layer_name,
                        save_dir=save_dir,
                        epoch=epoch,
                        include_metadata=True,
                        spikes_data=combined_spikes[layer_name],
                        metadata=metadata_for_save,
                        labels=combined_labels
                    )
                    saved_filepaths[layer_name] = filepath
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Fehler beim Speichern von {layer_name}: {e}")
        
        if verbose:
            print(f"✅ Monitoring abgeschlossen: {samples_collected} Samples für {len(layer_names)} Layer gespeichert")
        
        return saved_filepaths
        
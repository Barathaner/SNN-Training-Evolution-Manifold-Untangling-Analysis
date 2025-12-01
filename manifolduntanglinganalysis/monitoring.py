"""
SpikeActivityMonitor - Eine Library-Klasse zum automatischen Sammeln von Spike-Aktivitäten
aus SNN-Modellen ohne Modelländerungen.

Verwendet PyTorch Forward Hooks, um die Aktivitäten der LIF-Layer zu überwachen.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import os
from pathlib import Path


class SpikeActivityMonitor:
    """
    Überwacht und speichert Spike-Aktivitäten von LIF-Layern in einem SNN-Modell.
    
    Die Klasse registriert Forward Hooks auf spezifizierten Layern und sammelt
    automatisch die Spike- und Membranpotential-Aktivitäten während des Trainings.
    
    Beispiel:
        monitor = SpikeActivityMonitor(
            model=net,
            layer_names=['lif0', 'lif1', 'lif2', 'lif3'],
            save_dir='./activity_logs'
        )
        
        # Vor dem Training: Hooks registrieren
        monitor.start_monitoring()
        
        # Training...
        trainer.train_one_epoch_batched(...)
        
        # Nach der Epoche: Aktivitäten speichern
        monitor.save_epoch_activities(epoch=1)
        monitor.clear_activities()  # Für nächste Epoche
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        save_dir: str = "./activity_logs",
        collect_spikes: bool = True,
        collect_membrane: bool = True,
        max_samples_per_epoch: Optional[int] = None
    ):
        """
        Args:
            model: Das SNN-Modell, das überwacht werden soll
            layer_names: Liste der Layer-Namen, die überwacht werden sollen (z.B. ['lif0', 'lif1'])
            save_dir: Verzeichnis, in dem die Aktivitäten gespeichert werden
            collect_spikes: Ob Spike-Aktivitäten gesammelt werden sollen
            collect_membrane: Ob Membranpotentiale gesammelt werden sollen
            max_samples_per_epoch: Maximale Anzahl an Samples pro Epoche (None = alle)
        """
        self.model = model
        self.layer_names = layer_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.collect_spikes = collect_spikes
        self.collect_membrane = collect_membrane
        self.max_samples_per_epoch = max_samples_per_epoch
        
        # Speicher für gesammelte Aktivitäten
        # Struktur: {layer_name: {'spikes': [...], 'membrane': [...]}}
        self.activities: Dict[str, Dict[str, List[torch.Tensor]]] = {
            layer_name: {'spikes': [], 'membrane': []} 
            for layer_name in layer_names
        }
        
        # Hooks für die Layer
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Zähler für Samples in aktueller Epoche
        self.sample_count = 0
        
    def _hook_fn(self, layer_name: str, module, input, output):
        """
        Hook-Funktion, die bei jedem Forward-Pass des Layers aufgerufen wird.
        
        Für LIF-Layer: output ist ein Tupel (spike, membrane)
        """
        # LIF-Layer geben (spike, membrane) zurück
        if isinstance(output, tuple) and len(output) == 2:
            spike, membrane = output
            
            # Detach und auf CPU kopieren (spart GPU-Speicher)
            if self.collect_spikes:
                spike_detached = spike.detach().cpu()
                self.activities[layer_name]['spikes'].append(spike_detached)
            
            if self.collect_membrane:
                membrane_detached = membrane.detach().cpu()
                self.activities[layer_name]['membrane'].append(membrane_detached)
        else:
            # Fallback: Wenn Output kein Tupel ist, speichere es als Spike
            if self.collect_spikes:
                output_detached = output.detach().cpu()
                self.activities[layer_name]['spikes'].append(output_detached)
    
    def start_monitoring(self):
        """
        Registriert Forward Hooks auf den spezifizierten Layern.
        """
        self.clear_activities()  # Stelle sicher, dass Speicher leer ist
        
        for layer_name in self.layer_names:
            # Finde den Layer im Modell
            layer = dict(self.model.named_modules()).get(layer_name)
            
            if layer is None:
                raise ValueError(
                    f"Layer '{layer_name}' nicht im Modell gefunden. "
                    f"Verfügbare Layer: {list(dict(self.model.named_modules()).keys())}"
                )
            
            # Registriere Hook
            hook = layer.register_forward_hook(
                lambda m, i, o, name=layer_name: self._hook_fn(name, m, i, o)
            )
            self.hooks.append(hook)
        
        print(f"✅ Monitoring gestartet für Layer: {self.layer_names}")
    
    def stop_monitoring(self):
        """
        Entfernt alle registrierten Hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("🛑 Monitoring gestoppt")
    
    def clear_activities(self):
        """
        Löscht alle gesammelten Aktivitäten (für neue Epoche).
        """
        for layer_name in self.layer_names:
            self.activities[layer_name]['spikes'] = []
            self.activities[layer_name]['membrane'] = []
        self.sample_count = 0
    
    def get_activities(self, layer_name: Optional[str] = None) -> Dict:
        """
        Gibt die gesammelten Aktivitäten zurück.
        
        Args:
            layer_name: Wenn angegeben, nur Aktivitäten dieses Layers zurückgeben
        
        Returns:
            Dictionary mit Aktivitäten
        """
        if layer_name is not None:
            if layer_name not in self.activities:
                raise ValueError(f"Layer '{layer_name}' wird nicht überwacht.")
            return self.activities[layer_name]
        return self.activities
    
    def _stack_activities(self, activity_list: List[torch.Tensor], num_steps: int) -> torch.Tensor:
        """
        Stapelt Aktivitäten von einer Liste zu einem Tensor.
        
        Die Aktivitäten kommen als Liste von [B, features] Tensoren (ein pro Zeitschritt).
        Bei mehreren Batches: [Batch1_t0, Batch1_t1, ..., Batch1_tT, Batch2_t0, ...]
        Diese werden zu [total_B, T, features] gestapelt.
        
        Args:
            activity_list: Liste von Tensoren mit Shape [B, features]
            num_steps: Anzahl der Zeitschritte (T) pro Batch
        
        Returns:
            Tensor mit Shape [total_B, T, features] wobei total_B = Summe aller Batch-Größen
        """
        if len(activity_list) == 0:
            return torch.empty(0)
        
        total_entries = len(activity_list)
        batch_size = activity_list[0].shape[0]
        
        # Berechne Anzahl der vollständigen Batches
        num_batches = total_entries // num_steps
        
        if num_batches == 0:
            raise ValueError(
                f"Nicht genug Aktivitäten: {total_entries} Einträge, "
                f"aber {num_steps} Zeitschritte erwartet."
            )
        
        # Entferne unvollständige Batches
        if num_batches * num_steps < total_entries:
            activity_list = activity_list[:num_batches * num_steps]
        
        # Stack alle Aktivitäten: [total_entries, B, features]
        stacked = torch.stack(activity_list, dim=0)
        
        # Reshape zu [num_batches, num_steps, batch_size, features]
        stacked = stacked.view(num_batches, num_steps, batch_size, -1)
        
        # Permute zu [num_batches, batch_size, num_steps, features]
        stacked = stacked.permute(0, 2, 1, 3)
        
        # Reshape zu [num_batches * batch_size, num_steps, features]
        # Das ist die finale Form: [total_samples, T, features]
        stacked = stacked.reshape(num_batches * batch_size, num_steps, -1)
        
        return stacked
    
    def save_epoch_activities(
        self, 
        epoch: int, 
        num_steps: Optional[int] = None,
        save_format: str = "pt"
    ):
        """
        Speichert die gesammelten Aktivitäten einer Epoche.
        
        Args:
            epoch: Epochennummer
            num_steps: Anzahl der Zeitschritte pro Sample (wird automatisch erkannt, wenn None)
            save_format: Speicherformat ("pt" für PyTorch, "npz" für NumPy)
        """
        if num_steps is None:
            # Versuche num_steps aus dem Modell zu extrahieren
            if hasattr(self.model, 'num_steps'):
                num_steps = self.model.num_steps
            else:
                # Schätze aus der Anzahl der gesammelten Aktivitäten
                # Annahme: Alle Layer haben die gleiche Anzahl an Zeitschritten
                first_layer = self.layer_names[0]
                if len(self.activities[first_layer]['spikes']) > 0:
                    # Grobe Schätzung: Anzahl der Aktivitäten / erwartete Batches
                    num_steps = len(self.activities[first_layer]['spikes'])
                else:
                    raise ValueError(
                        "Konnte num_steps nicht bestimmen. Bitte explizit angeben."
                    )
        
        epoch_dir = self.save_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for layer_name in self.layer_names:
            layer_dir = epoch_dir / layer_name
            layer_dir.mkdir(exist_ok=True)
            
            # Spikes speichern
            if self.collect_spikes and len(self.activities[layer_name]['spikes']) > 0:
                spikes_stacked = self._stack_activities(
                    self.activities[layer_name]['spikes'], 
                    num_steps
                )
                
                if save_format == "pt":
                    spike_path = layer_dir / "spikes.pt"
                    torch.save(spikes_stacked, spike_path)
                    saved_files.append(str(spike_path))
                elif save_format == "npz":
                    spike_path = layer_dir / "spikes.npz"
                    import numpy as np
                    np.savez(spike_path, spikes=spikes_stacked.numpy())
                    saved_files.append(str(spike_path))
            
            # Membranpotentiale speichern
            if self.collect_membrane and len(self.activities[layer_name]['membrane']) > 0:
                membrane_stacked = self._stack_activities(
                    self.activities[layer_name]['membrane'], 
                    num_steps
                )
                
                if save_format == "pt":
                    membrane_path = layer_dir / "membrane.pt"
                    torch.save(membrane_stacked, membrane_path)
                    saved_files.append(str(membrane_path))
                elif save_format == "npz":
                    membrane_path = layer_dir / "membrane.npz"
                    import numpy as np
                    np.savez(membrane_path, membrane=membrane_stacked.numpy())
                    saved_files.append(str(membrane_path))
        
        print(f"💾 Aktivitäten für Epoch {epoch} gespeichert in: {epoch_dir}")
        return saved_files
    
    def __enter__(self):
        """Context Manager: Startet Monitoring beim Eintritt"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager: Stoppt Monitoring beim Verlassen"""
        self.stop_monitoring()
        return False


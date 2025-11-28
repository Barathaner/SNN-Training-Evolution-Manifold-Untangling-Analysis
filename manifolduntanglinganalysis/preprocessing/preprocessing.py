"""
Wiederverwendbare Preprocessing-Transformationen für SHD-Daten.
Entspricht dem Preprocessing aus dem Notebook für Manifold-Analysen.
"""

import numpy as np
import tonic.transforms as transforms
import numpy as np
import tonic.transforms as transforms
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class Downsample1D:
    def __init__(self, spatial_factor=0.1, target_size=None):
        self.spatial_factor = spatial_factor
        self.target_size = target_size

    def __call__(self, events):
        events = events.copy()
        # Skalierung mit mathematisch korrektem Runden
        scaled = np.round(events['x'] * self.spatial_factor)
        
        # Maximaler Wert berechnen: entweder aus target_size oder aus skalierten Events
        if self.target_size is not None:
            max_val = self.target_size - 1  # z.B. 70 Neuronen → Index 0-69
        else:
            # Dynamisch aus den ursprünglichen Events berechnen
            max_val = int(np.floor(events['x'].max() * self.spatial_factor))
        
        # Auf gültigen Bereich begrenzen
        events['x'] = np.clip(scaled, 0, max_val).astype(events['x'].dtype)
        return events


class SqrtTransform:
    """Wurzel-Transformation auf Frames (zur Normalisierung)"""
    def __init__(self, eps=0):
        self.eps = eps  # um sqrt(0) Probleme zu vermeiden
    
    def __call__(self, frames):
        frames = frames.astype(np.float32)
        return np.sqrt(frames + self.eps)


class TimeConvert:
    """
    Konvertiert Zeiteinheiten (z.B. von Mikrosekunden zu Millisekunden).
    Optional: Kann verwendet werden, falls Zeitkonvertierung benötigt wird.
    """
    def __init__(self, divisor=1000):
        self.divisor = divisor
    
    def __call__(self, events):
        events = events.copy()
        events["t"] = events["t"] // self.divisor
        return events



class DenoiseDBSCAN1D:
    """
    Entfernt Spikes, die nicht zu dichten zeitlich-räumlichen Clustern gehören.
    Berücksichtigt sowohl Zeit als auch Neuron-Position.
    Kompatibel mit tonic.transforms.
    
    Args:
        eps_time: Maximaler Zeitabstand innerhalb eines Clusters (in Zeiteinheiten)
        eps_spatial: Maximaler räumlicher Abstand (Neuron-Differenz)
        min_samples: Mindestanzahl an Events innerhalb eines Clusters
        use_spatial: Wenn True, wird 2D-Clustering (Zeit + Raum) verwendet
    """
    def __init__(self, eps_time=100, eps_spatial=5, min_samples=20, use_spatial=True):
        self.eps_time = eps_time
        self.eps_spatial = eps_spatial
        self.min_samples = min_samples
        self.use_spatial = use_spatial
    
    def __call__(self, events):
        """
        Args:
            events: structured numpy array mit 't', 'x' und optional 'p'
        
        Returns:
            Gefiltertes structured array
        """
        if len(events) == 0:
            return events
        
        if self.use_spatial:
            # 2D-Clustering: Zeit + räumliche Position
            # Normalisiere die Dimensionen, damit eps sinnvoll wirkt
            time_normalized = events["t"] / self.eps_time
            spatial_normalized = events["x"] / self.eps_spatial
            features = np.column_stack([time_normalized, spatial_normalized])
            eps = 1.0  # Weil wir normalisiert haben
        else:
            # 1D-Clustering: nur Zeit
            features = events["t"].reshape(-1, 1)
            eps = self.eps_time
        
        # DBSCAN anwenden
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(features)
        labels = clustering.labels_
        
        # -1 steht für Rauschen → wir behalten nur Clusterpunkte
        keep_mask = labels != -1
        return events[keep_mask]


class DenoiseKNN1D:
    """
    Behält nur Spikes, die genügend Nachbarn haben (schnelle Implementierung).
    
    Args:
        time_window: Zeitfenster (für 1D) oder Zeit-Divisor (für 2D).
        spatial_window: Raum-Divisor (nur für 2D).
        min_neighbors: Mindestanzahl an Nachbarevents.
        use_spatial: Wenn True, wird 2D-Radius-Suche verwendet (schnell).
                     Wenn False, wird 1D-Fenster-Suche verwendet (schnell).
    """
    def __init__(self, time_window=100, spatial_window=5, min_neighbors=5, use_spatial=True):
        self.time_window = time_window
        self.spatial_window = spatial_window
        self.min_neighbors = min_neighbors
        self.use_spatial = use_spatial
    
    def __call__(self, events):
        if len(events) <= self.min_neighbors:
            return events if len(events) > 0 and not self.use_spatial else np.empty(0, dtype=events.dtype)

        if self.use_spatial:
            # 🚀 SCHNELLE 2D-Implementierung (Radius-Suche, O(N log N))
            # Wir nutzen die gleiche Skalierungslogik wie DBSCAN
            time_norm = events["t"] / self.time_window
            spatial_norm = events["x"] / self.spatial_window
            features = np.column_stack([time_norm, spatial_norm])
            
            # Finde alle Nachbarn innerhalb eines Radius von 1.0 (im skalierten Raum)
            nn = NearestNeighbors(radius=1.0, n_jobs=-1)
            nn.fit(features)
            # radius_neighbors gibt Indizes-Listen zurück
            neighbor_indices = nn.radius_neighbors(features, return_distance=False)
            
            # Zähle Nachbarn (und ziehe 1 ab, um sich selbst zu ignorieren)
            neighbor_counts = np.array([len(indices) - 1 for indices in neighbor_indices])
            
            keep_mask = (neighbor_counts >= self.min_neighbors)
            return events[keep_mask]
            
        else:
            # 🚀 SCHNELLE 1D-Implementierung (Sortier-Suche, O(N log N))
            times = events["t"]
            sorted_indices = np.argsort(times)
            sorted_times = times[sorted_indices]
            
            keep_mask = np.zeros(len(events), dtype=bool)
            
            # Finde für jeden Punkt die Grenzen seines Fensters
            lower_bounds = sorted_times - self.time_window
            upper_bounds = sorted_times + self.time_window
            
            # Finde die Indizes dieser Grenzen im sortierten Array
            left_indices = np.searchsorted(sorted_times, lower_bounds, side='left')
            right_indices = np.searchsorted(sorted_times, upper_bounds, side='right')
            
            # Anzahl der Nachbarn = Differenz der Indizes (-1 für sich selbst)
            neighbor_counts = right_indices - left_indices - 1
            
            # Markiere die, die genug Nachbarn haben
            original_indices_to_keep = sorted_indices[neighbor_counts >= self.min_neighbors]
            keep_mask[original_indices_to_keep] = True
            
            return events[keep_mask]


def get_preprocessing(n_time_bins=80, target_neurons=70, original_neurons=700, 
                     fixed_duration=958007.0):
    """
    Gibt das Standard-Preprocessing für Manifold-Analysen zurück.
    
    Args:
        n_time_bins: Anzahl der Zeitbins (Standard: 80)
        target_neurons: Ziel-Anzahl Neuronen nach Downsampling
        original_neurons: Original-Anzahl Neuronen im Datensatz
        fixed_duration: Fixierte Sample-Dauer in μs (Standard: 958007 = 95. Perzentil)
                       Stellt konsistente zeitliche Auflösung sicher
    
    Returns:
        tonic.transforms.Compose Objekt
    """
    # Berechne Faktor und Sensorgröße dynamisch
    spatial_factor = float(target_neurons) / float(original_neurons)
    sensor_size = (target_neurons, 1, 1)
    
    # Berechne fixe Bin-Größe (time_window)
    time_window = float(fixed_duration) / float(n_time_bins)  # ≈ 11975 μs = 11.975 ms
    
    return transforms.Compose([
        # 1. Rauschen auf rohen Events entfernen
        # eps_time in Mikrosekunden: 100.000 μs = 100 ms
        DenoiseDBSCAN1D(eps_time=100000, eps_spatial=5, min_samples=20, use_spatial=True),
        
        # 2. Neuronen bündeln
        Downsample1D(
            spatial_factor=spatial_factor, 
            target_size=target_neurons
        ),
        
        # 3. Zu Frames umwandeln mit fixer zeitlicher Auflösung
        # Verwendet n_time_bins direkt für exakte Anzahl an Bins
        # - Jedes Bin = genau (fixed_duration / n_time_bins) μs
        # - Kürzere Samples: Leere Bins am Ende
        # - Längere Samples (>95%): Events nach fixed_duration werden ignoriert
        transforms.ToFrame(
            sensor_size=sensor_size, 
            n_time_bins=n_time_bins,
            include_incomplete=False
        ),
        
        # 4. Varianz stabilisieren
        #SqrtTransform()
    ])
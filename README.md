# SNN-Training-Evolution-Manifold-Untangling-Analysis
The learning process in Spiking Neural Networks can be geometrically explained as the untangling of the layer manifolds over epochs into linearly separable classes. This library provides a pipeline and an example to analyse your spiking data from the input and your snntorch model. It is based on Manifold Capacity, and dimensionalityreduction (UMAP)

## Abstract
The pace of development in Artificial Intelligence leads to increasingly complex
models. These models drive massive energy consumption, with data centers
requiring 415 TWh in 2024 and a projected demand of 945 TWh by 2030. Alter-
native theoretical models that are closer to the biological brain, such as Spiking
Neural Networks, run on neuromorphic hardware and are therefore more than
35 times more energy-efficient during training. This increase in efficiency comes
at the cost of complexity. Spiking Neural Networks are more complex due
to their additional usage of the temporal dimension in their calculations. To
explain their functioning, geometrical and topological indicators are analyzed.
The geometric indicators are properties of manifolds formed by each class, such
as radius, dimensionality, manifold capacity, and center correlation. Specifi-
cally, Mean-Field Theoretic Manifold Analysis is used to compute these metrics,
which provides a marker for the efficiency of untangling the global manifold
into separate class manifolds. Complementing these quantitative metrics, Uni-
form Manifold Approximation and Projection(UMAP) visualisations keep the
topological structure, which is qualitatively analyzed. UMAP is selected for its
ability to preserve global topological structures and its computational efficiency.
This work combines these indicators and geometrically explains the untangling
of class manifolds in a Deep Feed-Forward Spiking Neural Network trained to
classify the Spiking Heidelberg Digits dataset. The Neural Manifold Capacity
of the neural trajectories yields results similar to Artificial Neural Networks.
These neural trajectories also show a spatiotemporal dependency and converge
to a point of optimal clustering. The trained network exhibits a task-dependent
manifold throughout the layers, which emerges during training. Other meta-
data, such as speaker or gender, disappear. A correlation between accuracy and
Manifold Capacity provides evidence of the efficiency of each layer.





## Setup

Follow these steps to set up the project:

```bash
# Clone the repository
git clone https://github.com/Barathaner/SNN-Training-Evolution-Manifold-Untangling-Analysis.git-url
cd SNN-Training-Evolution-Manifold-Untangling-Analysis

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in editable mode
pip install -e .
```

or use the venv that is available and shipped.

The data folder contains the already monitored activity logs and the results of the mftma analysis for the network to test the plotting scripts in examples/ out.

In the examples folder you will find alot of scripts that have been used to generate plots or analysis.
the main.py contains a complete run from scratch. Be careful, this can take multiple hours depending on your system.

Therefore I split up the script in the data_analysis.ipynb and the data_harvesting.ipynb where you can see, what happens step by step in the main.py script and already see the outputs that it generated.

The data_harvesting.ipynb trains and records the neural network activity and plots the performance metrics.

The data_analysis.ipynb analyses and plots the mftma metrics.

There are various other scripts to create UMAP plots, Input dataset statistics, SHAP analysis, loss contours, mftma timestep analysis, pca_dimension estimations and preprocessing plots aswell as inference videos.

The web visualisation of the SHD dataset is a plotly dash app as app.py

# Pipeline-Konfiguration: Analyse Spiking Heidelberg Digits auf Feed-Forward Spiking Neural Network

## Übersicht

Diese Tabelle dokumentiert die vollständige Konfiguration der Pipeline für das Training und die Analyse eines Spiking Neural Networks auf dem SHD-Datensatz.

---

## 1. Dataset-Konfiguration

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Datensatz** | SHD (Spiking Heidelberg Digits) | Event-basierter Audiodatensatz |
| **Trainings-Split** | `train=True` | Verwendet Trainings-Split des Datensatzes |
| **Test-Split** | `train=False` | Verwendet Test-Split des Datensatzes |
| **Label-Bereich** | `range(0, 10)` | 10 Klassen (Ziffern 0-9) |
| **Batch-Größe (Training)** | `64` | Anzahl Samples pro Batch im Training |
| **Batch-Größe (Test)** | `64` | Anzahl Samples pro Batch im Test |
| **Drop Last** | `True` | Letzter unvollständiger Batch wird verworfen |
| **Shuffle** | `False` | Keine zufällige Reihenfolge (Standard) |

---

## 2. Preprocessing-Konfiguration

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Zeitbins (n_time_bins)** | `80` | Anzahl der diskreten Zeitschritte |
| **Original-Neuronen** | `700` | Anzahl Neuronen im Original-Datensatz |
| **Ziel-Neuronen (target_neurons)** | `350` | Anzahl Neuronen nach Downsampling (Faktor: 0.5) |
| **Fixe Dauer (fixed_duration)** | `958007.0 μs` | 95. Perzentil der Sample-Längen (~958 ms) |
| **Zeitfenster (time_window)** | `~11975 μs` | Berechnet als `fixed_duration / n_time_bins` (~12 ms pro Bin) |
| **Gaussian Sigma** | `1.0` | Standardabweichung für Gauß-Smoothing |
| **Trim Silence** | `False` | Stille am Anfang/Ende wird nicht entfernt |

### Preprocessing-Pipeline (Reihenfolge)

| Schritt | Komponente | Parameter |
|---------|-----------|-----------|
| **1. Denoising** | `DenoiseDBSCAN1D` | `eps_time=100000 μs` (100 ms), `eps_spatial=5`, `min_samples=20`, `use_spatial=True` |
| **2. Downsampling** | `Downsample1D` | `spatial_factor=0.5` (700 → 350 Neuronen), `target_size=350` |
| **3. Frame-Konvertierung** | `ToFrame` | `sensor_size=(350, 1, 1)`, `time_window=11975 μs`, `start_time=0.0`, `end_time=958007.0 μs`, `include_incomplete=True` |
| **4. Smoothing** | `GaussianSmoothing` | `sigma=1.0` |

---

## 3. Modell-Architektur

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Modell-Typ** | Feed-Forward Spiking Neural Network (sffnn_batched) | Batched Implementation |
| **Input-Dimension** | `350` | Entspricht Anzahl Neuronen nach Preprocessing |
| **Zeitschritte (num_steps)** | `80` | Anzahl der Zeitschritte im Forward-Pass |
| **Beta (LIF-Parameter)** | `0.9` | Leaky Integrate-and-Fire Decay-Faktor |

### Layer-Struktur

| Layer | Typ | Input → Output | Aktivierungsfunktion |
|-------|-----|----------------|---------------------|
| **Input** | - | `350` Neuronen | - |
| **fc0** | Linear | `350 → 350` | LIF0 (Leaky) |
| **lif0** | LIF Neuron | `350` Neuronen | `beta=0.9` |
| **fc1** | Linear | `350 → 128` | LIF1 (Leaky) |
| **lif1** | LIF Neuron | `128` Neuronen | `beta=0.9` |
| **fc2** | Linear | `128 → 64` | LIF2 (Leaky) |
| **lif2** | LIF Neuron | `64` Neuronen | `beta=0.9` |
| **fc3** | Linear | `64 → 10` | LIF3 (Leaky) |
| **lif3** | LIF Neuron | `10` Neuronen | `beta=0.9` |
| **Output** | Rate-Coded | `10` Klassen | Kumulative Spike-Summe über Zeit |

**Gesamt-Parameter**: ~350×350 + 350×128 + 128×64 + 64×10 = **122,500 + 44,800 + 8,192 + 640 = ~176,132 Parameter**

---

## 4. Training-Konfiguration

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Optimizer** | Adam | Adaptive Moment Estimation |
| **Learning Rate** | `5e-4` (0.0005) | Schrittweite für Parameter-Updates |
| **Loss-Funktion** | CrossEntropyLoss | Klassifikations-Loss |
| **Epochen (num_epochs)** | `10` | Anzahl Training-Epochen |
| **Device** | CUDA (falls verfügbar) / CPU | Automatische Geräteauswahl |
| **Seed** | `42` | Reproduzierbarkeit (PyTorch, NumPy, Random) |

### Metriken während Training

| Metrik | Training | Validation |
|--------|----------|------------|
| **Loss** | ✓ | ✓ |
| **Accuracy** | ✓ | ✓ |
| **Precision** | - | ✓ (weighted) |
| **Recall** | - | ✓ (weighted) |
| **F1-Score** | - | ✓ (weighted) |
| **AUC-ROC** | - | ✓ (weighted, multi-class OVR) |

---

## 5. Activity Monitoring

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Anzahl Samples** | `1000` | Samples die für Activity Logs gespeichert werden |
| **Überwachte Layer** | `['lif0', 'lif1', 'lif2', 'lif3']` | Alle 4 LIF-Layer werden überwacht |
| **Dataloader** | Test-Dataloader | Verwendet Test-Split für Monitoring |
| **Speicher-Format** | HDF5 (.h5) | Strukturiertes Format mit Events und Metadaten |
| **Dateinamen-Format** | `epoch_XXX_layername_spk_events.h5` | Strukturiert nach Epoche und Layer |

---

## 6. Manifold-Analyse-Konfiguration

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Analysierte Klassen** | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | Alle 10 Ziffern-Klassen |
| **Max Samples pro Klasse** | `100` | Maximale Anzahl Samples pro Klasse für Analyse |
| **Kappa (Margin)** | `0.0` | Margin-Größe für Manifold-Analyse |
| **n_t (Gaussian-Vektoren)** | `200` | Anzahl der Gaussian-Vektoren pro Manifold |
| **n_reps (Wiederholungen)** | `1` | Anzahl Wiederholungen für Korrelationsanalyse |
| **Output-Layer-Analyse** | Rate-Coded | Verwendet `analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded` |
| **Input-Daten-Analyse** | Standard | Verwendet `analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds` |

### Berechnete Metriken

| Metrik | Symbol | Beschreibung |
|--------|--------|--------------|
| **Capacity** | α_M | Durchschnittliche Manifold-Kapazität |
| **Radius** | R_M | Durchschnittlicher Manifold-Radius |
| **Dimension** | D_M | Durchschnittliche Manifold-Dimension |
| **Correlation** | - | Center-Korrelation zwischen Manifolds |
| **Optimal K** | K | Optimaler K-Wert für Analyse |
| **Per-Class Metriken** | - | Capacity, Radius, Dimension pro Klasse |

---

## 7. Activity Log Preprocessing

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **Zeitbins** | `80` | Entspricht num_steps des Modells |
| **Fixe Dauer** | `80` | Entspricht n_time_bins (diskrete Zeitschritte) |
| **Gaussian Sigma** | `1.0` | Standardabweichung für Smoothing |
| **Sensor-Größe** | `(num_neurons, 1, 1)` | Dynamisch basierend auf Layer-Größe |

### Activity Log Pipeline

| Schritt | Komponente | Parameter |
|---------|-----------|-----------|
| **1. Frame-Konvertierung** | `ToFrame` | `sensor_size=(num_neurons, 1, 1)`, `time_window=1.0`, `start_time=0.0`, `end_time=80`, `include_incomplete=True` |
| **2. Shape-Normalisierung** | `NormalizeFrameShape` | `num_neurons` (Layer-spezifisch) |
| **3. Smoothing** | `GaussianSmoothing` | `sigma=1.0` |

---

## 8. Ergebnisse-Speicherung

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **JSON-Struktur** | `results[epoch][layer]` | Verschachteltes Dictionary nach Epoche und Layer |
| **Gespeicherte Werte** | `capacity`, `radius`, `dimension` | Pro Epoche und Layer |
| **Datentyp-Konvertierung** | NumPy → Python native | `float64` → `float`, `int64` → `int` |
| **Speicher-Pfad** | `data/results/results.json` | Relativ zum Projekt-Root |
| **Plot-Speicher-Pfad** | `plots/` | Relativ zum Projekt-Root |

---

## 9. Zusammenfassung der Sample-Größen

| Phase | Anzahl Samples | Quelle |
|-------|----------------|--------|
| **Training** | Alle verfügbaren Samples (Label 0-9) | SHD Train-Split |
| **Test/Validation** | Alle verfügbaren Samples (Label 0-9) | SHD Test-Split |
| **Activity Monitoring** | `1000` Samples | Aus Test-Dataloader |
| **Manifold-Analyse (pro Klasse)** | Max. `100` Samples | Aus Activity Logs oder Test-Dataloader |
| **Manifold-Analyse (gesamt)** | Max. `1000` Samples (10 Klassen × 100) | Aus Activity Logs oder Test-Dataloader |

---

## 10. Technische Details

| Aspekt | Wert | Beschreibung |
|--------|------|-------------|
| **Framework** | PyTorch + snntorch | Deep Learning Framework |
| **Event-Processing** | Tonic | Event-basierte Datenverarbeitung |
| **Manifold-Analyse** | MFTMA (Mean-Field Theoretic Manifold Analysis) | Theoretische Manifold-Analyse |
| **Reproduzierbarkeit** | Seed = 42 | Für alle Zufallsgeneratoren |

---

*Erstellt am: $(date)*
*Pipeline: Spiking Heidelberg Digits auf Feed-Forward Spiking Neural Network*


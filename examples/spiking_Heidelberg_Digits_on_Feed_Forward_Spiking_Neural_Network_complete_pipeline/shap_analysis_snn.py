#!/usr/bin/env python3
"""
SHAP-Analyse für Spiking Neural Network (SNN).
Unterstützt zwei Methoden:
1. Captum Integrated Gradients (Standard, oft stabiler für PyTorch)
2. SHAP DeepExplainer (Alternative, für PyTorch-Modelle)

Visualisiert die Wichtigkeit von Input-Features (Neuronen × Zeit) für die Klassifizierung.
"""

import sys
import os
from pathlib import Path

# Füge Parent-Directory zum Path hinzu
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from datetime import datetime
from scipy.interpolate import interp1d

# Project imports
from preprocessing.dataloader import load_filtered_shd_dataloader
from preprocessing.preprocessing import get_preprocessing
from models.sffnn_batched import Net

# Captum für Integrated Gradients
try:
    from captum.attr import IntegratedGradients, GradientShap, Saliency
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️  Captum nicht verfügbar. Installiere mit: pip install captum")

# SHAP für DeepExplainer (PyTorch)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP nicht verfügbar. Installiere mit: pip install shap")


def create_prediction_wrapper(model, device):
    """
    Erstellt eine Wrapper-Funktion für die Vorhersage.
    Das SNN gibt [B, T, num_outputs] zurück, wir brauchen [B, num_outputs] für SHAP/Captum.
    
    Args:
        model: SNN-Modell
        device: torch.device
    
    Returns:
        prediction_fn: Funktion, die [B, T, Features] → [B, num_outputs] zurückgibt
    """
    def prediction_fn(x):
        """
        Args:
            x: Tensor mit Shape [B, T, Features] oder [B, T*Features] (flattened)
        
        Returns:
            logits: Tensor mit Shape [B, num_outputs]
        """
        model.eval()
        
        # Wenn x flattened ist, reshape zu [B, T, Features]
        if x.dim() == 2:
            # Annahme: x ist [B, T*Features]
            B = x.shape[0]
            # Wir müssen T und Features kennen - verwende Modell-Parameter
            T = model.num_steps
            num_features = model.fc0.in_features
            x = x.view(B, T, num_features)
        elif x.dim() == 3:
            # x ist bereits [B, T, Features]
            pass
        else:
            raise ValueError(f"Unerwartete Eingabe-Dimension: {x.dim()}")
        
        x = x.to(device)
        
        with torch.no_grad():
            spk_rec, _ = model(x)  # [B, T, num_outputs]
            # Rate Coding: Summe über Zeit
            spike_sums = spk_rec.sum(dim=1)  # [B, num_outputs]
        
        return spike_sums
    
    return prediction_fn


def create_model_wrapper_for_captum(model):
    """
    Erstellt einen Wrapper für das SNN-Modell, der für Captum kompatibel ist.
    Captum erwartet eine Funktion, die [B, T, Features] → [B, num_outputs] zurückgibt.
    
    Args:
        model: SNN-Modell
    
    Returns:
        wrapped_model: Wrapper-Funktion für Captum
    """
    class ModelWrapper(nn.Module):
        def __init__(self, snn_model):
            super().__init__()
            self.snn_model = snn_model
        
        def forward(self, x):
            """
            Args:
                x: Tensor [B, T, Features]
            
            Returns:
                logits: Tensor [B, num_outputs]
            """
            spk_rec, _ = self.snn_model(x)  # [B, T, num_outputs]
            # Rate Coding: Summe über Zeit
            spike_sums = spk_rec.sum(dim=1)  # [B, num_outputs]
            return spike_sums
    
    return ModelWrapper(model)


def compute_attributions_captum(model, inputs, targets, device, method='integrated_gradients', 
                                n_steps=50, baselines=None):
    """
    Berechnet Attributions mit Captum.
    
    Args:
        model: SNN-Modell
        inputs: Tensor [B, T, Features]
        targets: Tensor [B] mit Zielklassen
        device: torch.device
        method: 'integrated_gradients', 'gradient_shap', oder 'saliency'
        n_steps: Anzahl Schritte für Integrated Gradients
        baselines: Baseline-Tensor für GradientShap (optional)
    
    Returns:
        attributions: Tensor [B, T, Features] mit Attributions
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("Captum ist nicht verfügbar. Installiere mit: pip install captum")
    
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Erstelle Wrapper für Captum
    wrapped_model = create_model_wrapper_for_captum(model).to(device)
    wrapped_model.eval()
    
    # Erstelle Attributor basierend auf Methode
    if method == 'integrated_gradients':
        attributor = IntegratedGradients(wrapped_model, multiply_by_inputs=False)
        # Für Integrated Gradients: Baseline ist typischerweise Null
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        attributions = attributor.attribute(
            inputs,
            baselines=baselines,
            target=targets,
            n_steps=n_steps,
            return_convergence_delta=False
        )
    
    elif method == 'gradient_shap':
        attributor = GradientShap(wrapped_model)
        if baselines is None:
            # Erstelle mehrere Baselines durch zufällige Stichproben
            baselines = torch.cat([torch.zeros_like(inputs) for _ in range(5)], dim=0)
        attributions = attributor.attribute(
            inputs,
            baselines=baselines,
            target=targets,
            n_samples=5
        )
    
    elif method == 'saliency':
        attributor = Saliency(wrapped_model)
        attributions = attributor.attribute(inputs, target=targets)
    
    else:
        raise ValueError(f"Unbekannte Captum-Methode: {method}")
    
    return attributions


def compute_attributions_shap(model, inputs, targets, device, background_samples=None, 
                              n_samples=50):
    """
    Berechnet Attributions mit SHAP DeepExplainer (für PyTorch).
    
    Args:
        model: SNN-Modell
        inputs: Tensor [B, T, Features]
        targets: Tensor [B] mit Zielklassen
        device: torch.device
        background_samples: Background-Samples für SHAP [N, T, Features]
        n_samples: Anzahl Samples für SHAP (wird für DeepExplainer ignoriert)
    
    Returns:
        shap_values: SHAP-Werte als Tensor [B, T, Features]
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP ist nicht verfügbar. Installiere mit: pip install shap")
    
    model.eval()
    inputs = inputs.to(device)
    
    # Erstelle Modell-Wrapper für SHAP (nn.Module statt Funktion)
    wrapped_model = create_model_wrapper_for_captum(model).to(device)
    wrapped_model.eval()
    
    # Erstelle Background-Samples falls nicht vorhanden
    if background_samples is None:
        # Verwende Null-Tensor als Baseline
        background_samples = torch.zeros(1, inputs.shape[1], inputs.shape[2]).to(device)
    else:
        background_samples = background_samples.to(device)
    
    # SHAP DeepExplainer für PyTorch-Modelle
    # Hinweis: SHAP hat bekannte Probleme mit snnTorch's Leaky-Modulen
    # Wir versuchen es mit check_additivity=False, da SNNs nicht vollständig additiv sind
    print("⚠️  Hinweis: SHAP kann bei SNNs problematisch sein (snnTorch Leaky-Module).")
    print("   Empfehlung: Verwende --method captum für stabilere Ergebnisse.")
    
    try:
        explainer = shap.DeepExplainer(wrapped_model, background_samples)
        
        # Berechne SHAP-Werte mit check_additivity=False (SNNs sind nicht vollständig additiv)
        # Dies ist notwendig, da snnTorch's Leaky-Module nicht vollständig von SHAP unterstützt werden
        shap_values = explainer.shap_values(inputs, check_additivity=False)
        
        # Konvertiere zu Tensor falls nötig
        if isinstance(shap_values, list):
            # Liste von Arrays (eine pro Klasse) - konvertiere zu Tensor [B, T, F, num_classes]
            shap_values = np.array(shap_values)  # [num_classes, B, T, F]
            shap_values = np.transpose(shap_values, (1, 2, 3, 0))  # [B, T, F, num_classes]
            shap_values = torch.from_numpy(shap_values).to(device)
        elif isinstance(shap_values, np.ndarray):
            # Falls bereits Array: [B, T, F, num_classes] oder [B, T, F]
            shap_values = torch.from_numpy(shap_values).to(device)
        else:
            shap_values = torch.tensor(shap_values).to(device)
        
        # SHAP gibt [B, T, F, num_classes] zurück - wähle Attributions für vorhergesagte Klasse
        if shap_values.dim() == 4 and shap_values.shape[-1] > 1:
            # Extrahiere Attributions für die vorhergesagte Klasse
            with torch.no_grad():
                preds = wrapped_model(inputs)
                pred_classes = torch.argmax(preds, dim=1)  # [B]
            
            # Wähle Attributions für jede Sample's vorhergesagte Klasse
            # shap_values: [B, T, F, num_classes]
            # pred_classes: [B]
            selected_shap = []
            for i, pred_class in enumerate(pred_classes):
                selected_shap.append(shap_values[i, :, :, pred_class.item()])  # [T, F]
            shap_values = torch.stack(selected_shap, dim=0)  # [B, T, F]
        elif shap_values.dim() == 4:
            # Falls nur eine Klasse vorhanden, nimm die erste
            shap_values = shap_values[:, :, :, 0]  # [B, T, F]
        
        print(f"✅ SHAP DeepExplainer erfolgreich (mit check_additivity=False)")
        print(f"   Finale Attributions Shape: {shap_values.shape}")
        return shap_values
    
    except Exception as e:
        # Fallback: Verwende KernelExplainer mit kleiner Stichprobe
        print(f"⚠️  DeepExplainer fehlgeschlagen: {str(e)[:200]}")
        print("   Versuche KernelExplainer mit reduzierten Features...")
        
        try:
            # KernelExplainer ist langsamer, aber funktioniert mit allen Modellen
            # Reduziere die Anzahl Features für bessere Performance
            # Wir verwenden nur einen Teil der Zeitschritte
            B, T, F = inputs.shape
            
            # Verwende nur jeden 10. Zeitschritt für KernelExplainer (Performance)
            step_size = max(1, T // 50)  # Maximal 50 Zeitschritte
            reduced_inputs = inputs[:, ::step_size, :]  # [B, T_reduced, F]
            reduced_background = background_samples[:, ::step_size, :]
            
            print(f"   Reduziere Input von {T} auf {reduced_inputs.shape[1]} Zeitschritte für Performance")
            
            # Erstelle eine reduzierte Wrapper-Funktion
            def reduced_model_fn(x_flat):
                # x_flat: [n_samples, T_reduced * F]
                n_samples = x_flat.shape[0]
                x_reshaped = x_flat.reshape(n_samples, reduced_inputs.shape[1], F)
                # Fülle fehlende Zeitschritte mit Nullen
                x_full = torch.zeros(n_samples, T, F, device=device)
                x_full[:, ::step_size, :] = torch.from_numpy(x_reshaped).float().to(device)
                with torch.no_grad():
                    return wrapped_model(x_full).cpu().numpy()
            
            # Verwende KernelExplainer mit kleiner Stichprobe
            explainer = shap.KernelExplainer(
                reduced_model_fn,
                reduced_background[0].flatten().cpu().numpy().reshape(1, -1),
                max_evals=min(1000, 2 * reduced_inputs.shape[1] * F + 1)  # Genug für Permutation
            )
            
            # Berechne SHAP-Werte für reduziertes Input
            shap_values_reduced = explainer.shap_values(
                reduced_inputs[0].flatten().cpu().numpy().reshape(1, -1),
                nsamples=100  # Kleine Stichprobe für Performance
            )
            
            # Reshape zurück zu [T_reduced, F]
            shap_values_reduced = np.array(shap_values_reduced).reshape(reduced_inputs.shape[1], F)
            
            # Interpoliere zurück auf volle Größe [T, F]
            t_reduced = np.arange(0, T, step_size)
            t_full = np.arange(T)
            
            shap_values_full = np.zeros((T, F))
            for f in range(F):
                interp_fn = interp1d(t_reduced, shap_values_reduced[:, f], 
                                     kind='linear', fill_value='extrapolate')
                shap_values_full[:, f] = interp_fn(t_full)
            
            shap_values = torch.from_numpy(shap_values_full).unsqueeze(0).to(device)
            
            print("✅ SHAP KernelExplainer erfolgreich (mit Feature-Reduktion)")
            return shap_values
            
        except Exception as e2:
            # Letzter Fallback: Fehler mit Empfehlung
            raise RuntimeError(
                f"SHAP konnte nicht verwendet werden: {str(e2)[:200]}\n"
                "SHAP hat bekannte Kompatibilitätsprobleme mit snnTorch's Leaky-Modulen.\n"
                "Empfehlung: Verwende --method captum für stabilere Ergebnisse:\n"
                "  python manifolddatageneration/shap_analysis_snn.py --method captum"
            ) from e2


def plot_attributions_heatmap(attributions, inputs, labels, predictions, output_path, 
                               method_name='Integrated Gradients', sample_idx=0):
    """
    Erstellt Heatmap-Visualisierung der Attributions über Zeit und Neuronen.
    Visualisiert Spiking-Daten: Zeit (x-Achse) × Neuronen (y-Achse).
    
    Args:
        attributions: Tensor [B, T, Features] mit Attributions
        inputs: Tensor [B, T, Features] mit originalen Inputs (Spike Activity)
        labels: Tensor [B] mit Ground-Truth-Labels
        predictions: Tensor [B] mit Vorhersagen
        output_path: Pfad zum Speichern
        method_name: Name der verwendeten Methode
        sample_idx: Index des zu visualisierenden Samples
    """
    # Validiere Shapes
    if attributions.dim() != 3 or attributions.shape[0] <= sample_idx:
        raise ValueError(f"Unerwartete Attributions Shape: {attributions.shape}. Erwartet [B, T, Features]")
    if inputs.dim() != 3 or inputs.shape[0] <= sample_idx:
        raise ValueError(f"Unerwartete Input Shape: {inputs.shape}. Erwartet [B, T, Features]")
    
    # Wähle ein Sample
    attr_sample = attributions[sample_idx].detach().cpu().numpy()  # [T, Features]
    input_sample = inputs[sample_idx].detach().cpu().numpy()  # [T, Features]
    
    # Validiere, dass Shapes übereinstimmen
    if attr_sample.shape != input_sample.shape:
        raise ValueError(f"Shape-Mismatch: Attributions {attr_sample.shape} vs Input {input_sample.shape}")
    
    # Erstelle Figure mit Subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Heatmap der Attributions
    ax1 = axes[0]
    im1 = ax1.imshow(attr_sample.T, aspect='auto', cmap='RdBu_r', 
                     interpolation='nearest', vmin=-np.abs(attr_sample).max(), 
                     vmax=np.abs(attr_sample).max())
    ax1.set_xlabel('Zeitschritt (Time Bin)', fontsize=12)
    ax1.set_ylabel('Neuron Index', fontsize=12)
    ax1.set_title(f'{method_name} Attributions - Sample {sample_idx}\n'
                  f'Label: {labels[sample_idx].item()}, '
                  f'Prediction: {predictions[sample_idx].item()}', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Attribution Value')
    
    # 2. Heatmap der originalen Inputs (zum Vergleich)
    ax2 = axes[1]
    im2 = ax2.imshow(input_sample.T, aspect='auto', cmap='viridis', 
                     interpolation='nearest')
    ax2.set_xlabel('Zeitschritt (Time Bin)', fontsize=12)
    ax2.set_ylabel('Neuron Index', fontsize=12)
    ax2.set_title('Original Input (Spike Activity)', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='Spike Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Heatmap gespeichert: {output_path}")
    plt.close()


def plot_attributions_summary(attributions, inputs, labels, predictions, output_path,
                              method_name='Integrated Gradients', top_k=20):
    """
    Erstellt Summary-Plot: Wichtigste Neuronen und Zeitschritte.
    
    Args:
        attributions: Tensor [B, T, Features] mit Attributions
        inputs: Tensor [B, T, Features] mit originalen Inputs
        labels: Tensor [B] mit Ground-Truth-Labels
        predictions: Tensor [B] mit Vorhersagen
        output_path: Pfad zum Speichern
        method_name: Name der verwendeten Methode
        top_k: Anzahl der wichtigsten Features zum Anzeigen
    """
    # Aggregiere Attributions über alle Samples
    attr_abs = torch.abs(attributions).mean(dim=0)  # [T, Features] - Mittelwert über Batch
    
    # Summe über Zeit: Wichtigste Neuronen
    neuron_importance = attr_abs.sum(dim=0).detach().cpu().numpy()  # [Features]
    
    # Summe über Neuronen: Wichtigste Zeitschritte
    time_importance = attr_abs.sum(dim=1).detach().cpu().numpy()  # [T]
    
    # Erstelle Figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. Top-K Neuronen
    ax1 = axes[0]
    top_neuron_indices = np.argsort(neuron_importance)[-top_k:][::-1]
    top_neuron_values = neuron_importance[top_neuron_indices]
    
    ax1.barh(range(len(top_neuron_indices)), top_neuron_values)
    ax1.set_yticks(range(len(top_neuron_indices)))
    ax1.set_yticklabels([f'Neuron {i}' for i in top_neuron_indices])
    ax1.set_xlabel('Aggregierte Attribution (absolut)', fontsize=12)
    ax1.set_title(f'Top {top_k} Wichtigste Neuronen ({method_name})', fontsize=14)
    ax1.invert_yaxis()
    
    # 2. Wichtigkeit über Zeit
    ax2 = axes[1]
    ax2.plot(time_importance, linewidth=2)
    ax2.set_xlabel('Zeitschritt (Time Bin)', fontsize=12)
    ax2.set_ylabel('Aggregierte Attribution (absolut)', fontsize=12)
    ax2.set_title(f'Wichtigkeit über Zeit ({method_name})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary-Plot gespeichert: {output_path}")
    plt.close()


def prepare_shap_data_for_plots(attributions, inputs, n_top_features=50):
    """
    Bereitet Attributions für SHAP-Plots vor.
    Aggregiert über Zeit oder wählt Top-Features, da wir zu viele Features haben (T × F).
    
    Args:
        attributions: Tensor [B, T, F] mit Attributions
        inputs: Tensor [B, T, F] mit originalen Inputs
        n_top_features: Anzahl Top-Features zum Anzeigen
    
    Returns:
        shap_values_agg: Array [B, n_features] für SHAP-Plots
        feature_names: Liste von Feature-Namen
        inputs_agg: Array [B, n_features] mit aggregierten Inputs
    """
    B, T, F = attributions.shape
    
    # Option 1: Aggregiere über Zeit (Mittelwert) - zeigt wichtigste Neuronen
    attributions_agg_time = attributions.mean(dim=1)  # [B, F] - Mittelwert über Zeit
    inputs_agg_time = inputs.mean(dim=1)  # [B, F]
    
    # Option 2: Aggregiere über Neuronen (Mittelwert) - zeigt wichtigste Zeitschritte
    attributions_agg_neurons = attributions.mean(dim=2)  # [B, T] - Mittelwert über Neuronen
    inputs_agg_neurons = inputs.mean(dim=2)  # [B, T]
    
    # Option 3: Wähle Top-Features (absoluter Mittelwert über Batch)
    attributions_abs_mean = torch.abs(attributions).mean(dim=0)  # [T, F]
    top_indices = torch.topk(attributions_abs_mean.flatten(), n_top_features).indices
    
    # Konvertiere zu Feature-Namen und extrahiere Werte
    top_features_time = []
    top_features_neuron = []
    top_features_values = []
    top_inputs_values = []
    
    for idx in top_indices:
        t_idx = idx.item() // F
        f_idx = idx.item() % F
        top_features_time.append(t_idx)
        top_features_neuron.append(f_idx)
        top_features_values.append(attributions[:, t_idx, f_idx].cpu().numpy())  # [B]
        top_inputs_values.append(inputs[:, t_idx, f_idx].cpu().numpy())  # [B]
    
    # Stack zu [B, n_top_features]
    shap_values_top = np.stack(top_features_values, axis=1)  # [B, n_top_features]
    inputs_top = np.stack(top_inputs_values, axis=1)  # [B, n_top_features]
    
    # Feature-Namen (lesbarer: "Time 123, Neuron 45")
    feature_names = [f"Time {t}, Neuron {n}" for t, n in zip(top_features_time, top_features_neuron)]
    
    return shap_values_top, feature_names, inputs_top


def plot_shap_visualizations(attributions, inputs, labels, predictions, model, device,
                             output_dir, method_name='SHAP', n_samples=5):
    """
    Erstellt SHAP-Plots: Force Plot, Bar Plot, Beeswarm Plot.
    
    Args:
        attributions: Tensor [B, T, F] mit Attributions
        inputs: Tensor [B, T, F] mit originalen Inputs
        labels: Tensor [B] mit Ground-Truth-Labels
        predictions: Tensor [B] mit Vorhersagen
        model: Modell (für Base-Value-Berechnung)
        device: torch.device
        output_dir: Output-Verzeichnis
        method_name: Name der Methode
        n_samples: Anzahl Samples
    """
    if not SHAP_AVAILABLE:
        return
    
    print("   Bereite Daten für SHAP-Plots vor...")
    
    # Bereite Daten vor (wähle Top-Features für bessere Visualisierung)
    shap_values_agg, feature_names, inputs_agg = prepare_shap_data_for_plots(
        attributions, inputs, n_top_features=50
    )
    
    # Berechne Base-Value (Durchschnittliche Vorhersage über Baseline)
    model.eval()
    with torch.no_grad():
        # Verwende Null-Input als Baseline (typisch für SNNs)
        baseline = torch.zeros_like(inputs[:1])
        spk_rec, _ = model(baseline)
        spike_sums = spk_rec.sum(dim=1)  # [1, num_outputs]
        
        # Base-Value ist die Vorhersage für Baseline (Null-Input)
        # Für Multi-Class: Verwende den Wert für die vorhergesagte Klasse
        # Oder den Durchschnitt über alle Klassen
        base_value = spike_sums.mean().item()
        
        # Alternative: Berechne Base-Value für jede vorhergesagte Klasse
        base_values_per_class = spike_sums[0].cpu().numpy()  # [num_classes]
    
    # Erstelle SHAP Explanation-Objekt für Plots
    # SHAP erwartet ein Explanation-Objekt mit bestimmten Attributen
    class SimpleExplanation:
        def __init__(self, values, data, base_values, feature_names):
            self.values = values  # [n_samples, n_features]
            self.data = data  # [n_samples, n_features] - Original-Inputs
            self.base_values = base_values  # [n_samples] - Base-Values
            self.feature_names = feature_names  # Liste von Feature-Namen
    
    # Berechne Base-Values für jedes Sample
    # Verwende den Base-Value für die vorhergesagte Klasse jedes Samples
    base_values = []
    for i in range(n_samples):
        pred_class = predictions[i].item()
        base_values.append(base_values_per_class[pred_class])
    base_values = np.array(base_values)
    
    explanation = SimpleExplanation(
        values=shap_values_agg,
        data=inputs_agg,
        base_values=base_values,
        feature_names=feature_names
    )
    
    # 1. Force Plot für einzelne Samples
    print("   Erstelle Force Plots...")
    for i in range(min(n_samples, 3)):  # Maximal 3 Force Plots
        try:
            force_path = output_dir / f"shap_force_sample_{i}_{method_name.lower().replace(' ', '_')}.png"
            
            # Erstelle Force Plot mit matplotlib (SHAP's force_plot ist interaktiv)
            fig, ax = plt.subplots(figsize=(14, 4))
            
            # Sortiere Features nach absoluter Wichtigkeit
            sample_shap = shap_values_agg[i]
            sample_input = inputs_agg[i]
            sorted_indices = np.argsort(np.abs(sample_shap))[::-1][:20]  # Top 20
            
            # Erstelle horizontale Balken
            y_pos = np.arange(len(sorted_indices))
            values = sample_shap[sorted_indices]
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[idx] for idx in sorted_indices], fontsize=8)
            ax.set_xlabel('SHAP Value', fontsize=12)
            ax.set_title(f'Force Plot - Sample {i}\n'
                        f'Label: {labels[i].item()}, Prediction: {predictions[i].item()}\n'
                        f'Base Value: {base_value:.3f}, Output: {base_value + sample_shap.sum():.3f}',
                        fontsize=11)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(force_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Force Plot Sample {i} gespeichert")
        except Exception as e:
            print(f"      ⚠️  Force Plot Sample {i} fehlgeschlagen: {e}")
    
    # 2. Bar Plot (Absolute Mean SHAP)
    print("   Erstelle Bar Plot...")
    try:
        bar_path = output_dir / f"shap_bar_{method_name.lower().replace(' ', '_')}.png"
        
        # Berechne absolute Mittelwerte über alle Samples
        mean_abs_shap = np.abs(shap_values_agg).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:30]  # Top 30
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(sorted_indices))
        values = mean_abs_shap[sorted_indices]
        
        ax.barh(y_pos, values, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[idx] for idx in sorted_indices], fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title(f'Bar Plot - Top 30 Features ({method_name})', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Bar Plot gespeichert")
    except Exception as e:
        print(f"      ⚠️  Bar Plot fehlgeschlagen: {e}")
    
    # 3. Beeswarm Plot
    print("   Erstelle Beeswarm Plot...")
    try:
        beeswarm_path = output_dir / f"shap_beeswarm_{method_name.lower().replace(' ', '_')}.png"
        
        # Wähle Top-Features für Beeswarm (zu viele Features = unlesbar)
        mean_abs_shap = np.abs(shap_values_agg).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:20]  # Top 20
        
        # Erstelle Beeswarm-ähnliche Visualisierung
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Für jedes Feature: Scatter-Plot der SHAP-Werte
        for idx, feat_idx in enumerate(top_indices):
            y_pos = idx
            shap_vals = shap_values_agg[:, feat_idx]
            input_vals = inputs_agg[:, feat_idx]
            
            # Farbe basierend auf Input-Wert
            scatter = ax.scatter(shap_vals, [y_pos] * len(shap_vals), 
                               c=input_vals, cmap='RdBu_r', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([feature_names[idx] for idx in top_indices], fontsize=9)
        ax.set_xlabel('SHAP Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Beeswarm Plot - Top 20 Features ({method_name})', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Input Value', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Beeswarm Plot gespeichert")
    except Exception as e:
        print(f"      ⚠️  Beeswarm Plot fehlgeschlagen: {e}")


def run_shap_analysis(
    model_path='./model_export/model_weights.pth',
    method='captum',  # 'captum' oder 'shap'
    captum_method='integrated_gradients',  # 'integrated_gradients', 'gradient_shap', 'saliency'
    n_samples=5,
    n_steps=50,
    output_dir='./shap_analysis',
    device=None
):
    """
    Hauptfunktion für SHAP-Analyse des SNN.
    
    Args:
        model_path: Pfad zu den Modell-Gewichten
        method: 'captum' oder 'shap'
        captum_method: Methode für Captum ('integrated_gradients', 'gradient_shap', 'saliency')
        n_samples: Anzahl Samples zum Analysieren
        n_steps: Anzahl Schritte für Integrated Gradients
        output_dir: Output-Verzeichnis
        device: torch.device (optional)
    """
    print("=" * 80)
    print("SHAP-ANALYSE FÜR SPIKING NEURAL NETWORK")
    print("=" * 80)
    
    # Device Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Erstelle Output-Verzeichnis
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # MODELL LADEN
    # ============================================================================
    print(f"\n📦 Lade Modell von {model_path}...")
    model = Net(
        num_inputs=350,
        num_hidden1=128,
        num_hidden2=64,
        num_outputs=10,
        num_steps=500,
        beta=0.9
    ).to(device)
    
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Modell erfolgreich geladen")
    else:
        print(f"⚠️  Warnung: Modell-Pfad {model_path} existiert nicht.")
        print("   Verwende zufällig initialisiertes Modell (nur für Tests)")
    
    model.eval()
    
    # ============================================================================
    # DATEN LADEN
    # ============================================================================
    print(f"\n📂 Lade Testdaten...")
    transform = get_preprocessing(
        n_time_bins=500,
        target_neurons=350,
        original_neurons=700,
        fixed_duration=958007.0
    )
    
    test_dataloader = load_filtered_shd_dataloader(
        label_range=range(0, 10),
        transform=transform,
        train=False,
        batch_size=1,  # Ein Sample pro Batch für SHAP
        shuffle=False,
        drop_last=False
    )
    
    # Sammle Samples
    print(f"\n🔍 Sammle {n_samples} Samples für Analyse...")
    samples = []
    labels_list = []
    
    for i, (events, labels) in enumerate(test_dataloader):
        if i >= n_samples:
            break
        
        # Events von [B, T, 1, Features] → [B, T, Features]
        if events.ndim == 4:
            events = events.squeeze(2)
        
        samples.append(events)
        labels_list.append(labels)
    
    inputs = torch.cat(samples, dim=0).to(device)  # [n_samples, T, Features]
    labels = torch.cat(labels_list, dim=0).to(device)  # [n_samples]
    
    print(f"✅ {len(samples)} Samples geladen")
    print(f"   Input Shape: {inputs.shape}")
    
    # ============================================================================
    # VORHERSAGEN
    # ============================================================================
    print(f"\n🔮 Berechne Vorhersagen...")
    with torch.no_grad():
        spk_rec, _ = model(inputs)
        spike_sums = spk_rec.sum(dim=1)  # [n_samples, num_outputs]
        predictions = torch.argmax(spike_sums, dim=1)
    
    accuracy = (predictions == labels).float().mean().item()
    print(f"✅ Accuracy auf Test-Samples: {accuracy:.2%}")
    
    # ============================================================================
    # ATTRIBUTIONS BERECHNEN
    # ============================================================================
    print(f"\n🧮 Berechne Attributions mit {method.upper()}...")
    
    if method == 'captum':
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist nicht verfügbar. Installiere mit: pip install captum")
        
        # Erstelle Baseline (Null-Tensor)
        baselines = torch.zeros_like(inputs)
        
        # Berechne Attributions
        attributions = compute_attributions_captum(
            model=model,
            inputs=inputs,
            targets=predictions,  # Verwende Vorhersagen als Targets
            device=device,
            method=captum_method,
            n_steps=n_steps,
            baselines=baselines
        )
        
        method_name = f"Captum {captum_method.replace('_', ' ').title()}"
    
    elif method == 'shap':
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP ist nicht verfügbar. Installiere mit: pip install shap")
        
        # Erstelle Background-Samples
        background_samples = torch.zeros(1, inputs.shape[1], inputs.shape[2]).to(device)
        
        # Berechne Attributions
        attributions = compute_attributions_shap(
            model=model,
            inputs=inputs,
            targets=predictions,
            device=device,
            background_samples=background_samples,
            n_samples=n_steps
        )
        
        method_name = "SHAP DeepExplainer"
    
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
    
    print(f"✅ Attributions berechnet. Shape: {attributions.shape}")
    
    # ============================================================================
    # VISUALISIERUNGEN
    # ============================================================================
    print(f"\n📊 Erstelle Visualisierungen...")
    
    # 1. Heatmap für jedes Sample
    for i in range(min(n_samples, 5)):  # Maximal 5 Samples visualisieren
        heatmap_path = output_dir / f"heatmap_sample_{i}_{method}_{captum_method}.png"
        plot_attributions_heatmap(
            attributions=attributions,
            inputs=inputs,
            labels=labels,
            predictions=predictions,
            output_path=heatmap_path,
            method_name=method_name,
            sample_idx=i
        )
    
    # 2. Summary-Plot
    summary_path = output_dir / f"summary_{method}_{captum_method}.png"
    plot_attributions_summary(
        attributions=attributions,
        inputs=inputs,
        labels=labels,
        predictions=predictions,
        output_path=summary_path,
        method_name=method_name,
        top_k=20
    )
    
    # 3. SHAP-Plots (wenn SHAP verfügbar ist - funktioniert auch für Captum-Attributions)
    if SHAP_AVAILABLE:
        print("\n📊 Erstelle SHAP-Plots (Force Plot, Bar Plot, Beeswarm Plot)...")
        try:
            plot_shap_visualizations(
                attributions=attributions,
                inputs=inputs,
                labels=labels,
                predictions=predictions,
                model=model,
                device=device,
                output_dir=output_dir,
                method_name=method_name,
                n_samples=n_samples
            )
        except Exception as e:
            print(f"⚠️  SHAP-Plots konnten nicht erstellt werden: {e}")
            print("   Heatmaps und Summary-Plots sind weiterhin verfügbar.")
    
    # 4. Speichere Attributions als NumPy-Array
    attributions_np = attributions.detach().cpu().numpy()
    np.save(output_dir / f"attributions_{method}_{captum_method}.npy", attributions_np)
    print(f"✅ Attributions gespeichert: {output_dir / f'attributions_{method}_{captum_method}.npy'}")
    
    print(f"\n{'='*80}")
    print("SHAP-ANALYSE ABGESCHLOSSEN")
    print(f"{'='*80}")
    print(f"\n📁 Ergebnisse gespeichert in: {output_dir}")
    print(f"   - Heatmaps: heatmap_sample_*.png")
    print(f"   - Summary: summary_{method}_{captum_method}.png")
    if SHAP_AVAILABLE:
        print(f"   - SHAP-Plots: shap_force_*.png, shap_bar.png, shap_beeswarm.png")
    print(f"   - Attributions: attributions_{method}_{captum_method}.npy")


def main():
    parser = argparse.ArgumentParser(
        description='SHAP-Analyse für Spiking Neural Network'
    )
    parser.add_argument('--model-path', type=str, default='./model_export/model_weights.pth',
                       help='Pfad zu den Modell-Gewichten')
    parser.add_argument('--method', type=str, default='captum',
                       choices=['captum', 'shap'],
                       help='Methode: captum (Standard) oder shap')
    parser.add_argument('--captum-method', type=str, default='integrated_gradients',
                       choices=['integrated_gradients', 'gradient_shap', 'saliency'],
                       help='Captum-Methode (nur wenn --method=captum)')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Anzahl Samples zum Analysieren')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Anzahl Schritte für Integrated Gradients / SHAP')
    parser.add_argument('--output-dir', type=str, default='./shap_analysis',
                       help='Output-Verzeichnis')
    
    args = parser.parse_args()
    
    run_shap_analysis(
        model_path=args.model_path,
        method=args.method,
        captum_method=args.captum_method,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()


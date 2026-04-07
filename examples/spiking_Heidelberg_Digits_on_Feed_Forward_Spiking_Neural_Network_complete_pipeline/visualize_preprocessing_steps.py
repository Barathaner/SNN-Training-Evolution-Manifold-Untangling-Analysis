"""
Script to visualize preprocessing transformations step by step.
Loads a sample from the SHD dataset and creates a plot after each transformation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tonic
import tonic.transforms as transforms
from pathlib import Path

# Projekt-Imports
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms

# Projekt-Root bestimmen
project_root = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(project_root, "data", "input")
output_dir = os.path.join(project_root, "plots", "preprocessing_steps")
os.makedirs(output_dir, exist_ok=True)

# Preprocessing-Parameter (aus Komplette_Konfiguration.md)
n_time_bins = 80
target_neurons = 350
original_neurons = 700
fixed_duration = 958007.0
gaussian_sigma = 1.0

def plot_raster_from_events(events, label, title, save_path, step_num):
    """
    Creates a raster plot from raw events.
    
    Args:
        events: structured numpy array with 't' and 'x'
        label: Sample label
        title: Plot title
        save_path: Path to save
        step_num: Step number for filename
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='w')
    
    if len(events) > 0:
        # Convert time from μs to ms for better readability
        times_ms = events['t'] / 1000.0
        neurons = events['x']
        
        # Raster Plot with larger points for better visibility
        ax.scatter(times_ms, neurons, s=3.0, marker='|', color='black', alpha=0.7, linewidths=1.5)
        
        # Statistics
        num_spikes = len(events)
        duration_ms = (events['t'].max() - events['t'].min()) / 1000.0 if len(events) > 0 else 0
        num_neurons = len(np.unique(neurons)) if len(events) > 0 else 0
        
        # Large, clear labels in English
        ax.set_xlabel('Time (ms)', fontsize=28, fontweight='bold')
        ax.set_ylabel('Neuron Index', fontsize=28, fontweight='bold')
        ax.set_title(f'{title}\nLabel: {label} | Spikes: {num_spikes} | Duration: {duration_ms:.1f} ms | Neurons: {num_neurons}', 
                     fontsize=24, fontweight='bold', pad=20)
        ax.tick_params(labelsize=20)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No events after this transformation', 
                ha='center', va='center', fontsize=32, transform=ax.transAxes, fontweight='bold')
        ax.set_title(f'{title}\nLabel: {label}', fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {save_path}")
    plt.close()

def plot_frames_heatmap(frames, label, title, save_path, step_num):
    """
    Creates a heatmap plot from frames.
    
    Args:
        frames: numpy array with Shape (n_time_bins, n_neurons) or similar
        label: Sample label
        title: Plot title
        save_path: Path to save
        step_num: Step number for filename
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='w')
    
    # Normalize shape to (n_time_bins, n_neurons)
    if frames.ndim == 4:
        frames = frames[:, 0, 0, :]
    elif frames.ndim == 3:
        if frames.shape[1] == 1:
            frames = frames[:, 0, :]
        elif frames.shape[2] == 1:
            frames = frames[:, :, 0]
        else:
            frames = frames.reshape(frames.shape[0], -1)
    elif frames.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected frame shape: {frames.shape}")
    
    # Heatmap
    im = ax.imshow(frames.T, aspect='auto', cmap='hot', origin='lower', interpolation='nearest')
    
    # Colorbar with larger text
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spike Count / Activity', fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
    
    # Large, clear labels in English
    ax.set_xlabel('Time Bin', fontsize=28, fontweight='bold')
    ax.set_ylabel('Neuron Index', fontsize=28, fontweight='bold')
    ax.set_title(f'{title}\nLabel: {label} | Shape: {frames.shape}', 
                 fontsize=24, fontweight='bold', pad=20)
    ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {save_path}")
    plt.close()

def main():
    print("="*80)
    print("Visualization of Preprocessing Transformations")
    print("="*80)
    
    # 1. Load raw sample from SHD dataset (without transformation)
    print("\n📊 Loading raw sample from SHD dataset...")
    dataset = tonic.datasets.SHD(save_to=data_path, train=False, transform=None)
    
    # Find a sample with label 0-9
    sample_idx = None
    for i in range(len(dataset)):
        events, label = dataset[i]
        if label < 10:
            sample_idx = i
            break
    
    if sample_idx is None:
        raise ValueError("No sample with label 0-9 found")
    
    events_raw, label = dataset[sample_idx]
    print(f"✅ Sample loaded - Index: {sample_idx}, Label: {label}")
    print(f"   Number of events: {len(events_raw)}")
    
    # 2. Plot 0: Raw Events (without transformation)
    print("\n🎨 Creating Plot 0: Raw Events...")
    plot_raster_from_events(
        events_raw, 
        label, 
        "Step 0: Raw Events (No Transformation)",
        os.path.join(output_dir, "step_00_raw_events.png"),
        0
    )
    
    # 3. Step 1: DenoiseDBSCAN1D
    print("\n🔧 Step 1: DenoiseDBSCAN1D...")
    denoise_transform = datatransforms.DenoiseDBSCAN1D(
        eps_time=100000, 
        eps_spatial=5, 
        min_samples=20, 
        use_spatial=True
    )
    events_denoised = denoise_transform(events_raw.copy())
    print(f"   Events before: {len(events_raw)}, after: {len(events_denoised)}")
    
    plot_raster_from_events(
        events_denoised,
        label,
        "Step 1: After DenoiseDBSCAN1D",
        os.path.join(output_dir, "step_01_denoised.png"),
        1
    )
    
    # 4. Step 2: Downsample1D
    print("\n🔧 Step 2: Downsample1D...")
    spatial_factor = float(target_neurons) / float(original_neurons)
    downsample_transform = datatransforms.Downsample1D(
        spatial_factor=spatial_factor,
        target_size=target_neurons
    )
    events_downsampled = downsample_transform(events_denoised.copy())
    print(f"   Neurons before: {len(np.unique(events_denoised['x']))}, after: {len(np.unique(events_downsampled['x']))}")
    
    plot_raster_from_events(
        events_downsampled,
        label,
        "Step 2: After Downsample1D (700 → 350 Neurons)",
        os.path.join(output_dir, "step_02_downsampled.png"),
        2
    )
    
    # 5. Step 3: ToFrame
    print("\n🔧 Step 3: ToFrame...")
    time_window = float(fixed_duration) / float(n_time_bins)
    sensor_size = (target_neurons, 1, 1)
    toframe_transform = transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=time_window,
        start_time=0.0,
        end_time=fixed_duration,
        include_incomplete=True
    )
    frames = toframe_transform(events_downsampled)
    print(f"   Frames Shape: {frames.shape}")
    
    plot_frames_heatmap(
        frames,
        label,
        "Step 3: After ToFrame (Events → Frames)",
        os.path.join(output_dir, "step_03_frames.png"),
        3
    )
    
    # 6. Step 4: GaussianSmoothing
    print("\n🔧 Step 4: GaussianSmoothing...")
    smoothing_transform = datatransforms.GaussianSmoothing(sigma=gaussian_sigma)
    frames_smoothed = smoothing_transform(frames)
    print(f"   Frames Shape: {frames_smoothed.shape}")
    
    plot_frames_heatmap(
        frames_smoothed,
        label,
        f"Step 4: After GaussianSmoothing (σ={gaussian_sigma})",
        os.path.join(output_dir, "step_04_smoothed.png"),
        4
    )
    
    print("\n" + "="*80)
    print(f"✅ All plots saved in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

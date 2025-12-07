import torch.nn as nn
import os
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
from manifolduntanglinganalysis.training import Trainer
import models.sffnn_batched as sffnn_batched
from manifolduntanglinganalysis.ActivityMonitor import ActivityMonitor
from manifolduntanglinganalysis.preprocessing.metadata_extractor import SHDMetadataExtractor
import manifolduntanglinganalysis.analysis.intrinsic_dimension as id_analysis
from manifolduntanglinganalysis.metrics.mean_field_theoretic_manifold_analysis_wrapper import analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds, plot_manifold_metrics_over_epochs
import numpy as np
import random
import torch
import h5py
from tonic.transforms import ToFrame
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    # Load the dataset in the data/input folder from the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "data", "input")
    transform = datatransforms.get_preprocessing(
        n_time_bins=80,
        target_neurons=350,
        original_neurons=700,
        fixed_duration=958007.0
    )


    # Data loading
    test_dataloader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10),
        data_path=data_path,
        transform=transform, 
        train=False, 
        batch_size=64,
        num_samples=512
    )

    # results_input = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
    #     dataloader=test_dataloader,
    #     labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     max_samples_per_class=100,
    #     kappa=0.0,
    #     n_t=200,
    #     n_reps=1,
    #     verbose=True
    # )
    results_input = {
        'capacity': 0.0069,
        'radius': 1.8510,
        'dimension': 187.0195,
        # Ich füge die anderen Werte aus deinem Text auch hinzu, falls du sie brauchst:
        'correlation': 0.5949,
        'optimal_k': 2
    }
    print(f"Capacity: {results_input['capacity']:.4f}")
    print(f"Radius: {results_input['radius']:.4f}")
    print(f"Dimension: {results_input['dimension']:.4f}")
    #construct path of all activity logs
    activity_logs_path = os.path.join(project_root, "data", "activity_logs")
    activity_logs = os.listdir(activity_logs_path)
    
    # Sortiere Activity Logs nach Epoche und Layer
    # Format: epoch_XXX_layername_spk_events.h5
    def sort_key(filename):
        import re
        match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', filename)
        if match:
            epoch = int(match.group(1))
            layer = match.group(2)
            # Sortiere zuerst nach Epoche, dann nach Layer
            return (epoch, layer)
        return (999, 'zzz')  # Unbekannte Dateien ans Ende
    
    activity_logs = sorted(activity_logs, key=sort_key)
    
    results = []
    pca_intdims = []
    
    for activity_log in activity_logs:
        # Lade Activity Log und erstelle Transform mit korrekter sensor_size
        activity_log_path = os.path.join(activity_logs_path, activity_log)
        
        # Lese num_features aus H5-Datei für sensor_size
        with h5py.File(activity_log_path, 'r') as f:
            if 'num_features' in f.attrs:
                num_neurons = int(f.attrs['num_features'])
            else:
                # Fallback: Versuche aus Events zu bestimmen
                if 'events' in f and len(f['events']) > 0:
                    first_sample_key = sorted(f['events'].keys())[0]
                    events = f['events'][first_sample_key][:]
                    num_neurons = int(events['x'].max() + 1) if len(events) > 0 else 128
                else:
                    num_neurons = 128  # Standard-Fallback
        
        # Erstelle Transform mit korrekter sensor_size (Format: (neurons, height, width))
        activity_log_transform = ToFrame(
            sensor_size=(num_neurons, 1, 1),
            n_time_bins=80,
            include_incomplete=True
        )
        
        # Lade Activity Log mit Transform
        activity_log_dataloader = dataloader.load_activity_log(
            activity_log_path=activity_log_path, 
            transform=activity_log_transform
        )
        print(f"Analyzing activity log: {activity_log}")
        current_result = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
            dataloader=activity_log_dataloader,
            labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            max_samples_per_class=100,
            kappa=0.0,
            n_t=200,
            n_reps=1,
            verbose=True
        )

        # 2. Auf die temporäre Variable zugreifen für den Print
        print(f"Capacity: {current_result['capacity']:.4f}")
        print(f"Radius: {current_result['radius']:.4f}")
        print(f"Dimension: {current_result['dimension']:.4f}")

        # 3. Das Ergebnis zur Liste hinzufügen
        results.append(current_result)
    plot_manifold_metrics_over_epochs(results, activity_logs_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))


    # Data loading
    # train_dataloader = dataloader.load_filtered_shd_dataloader(
    #     label_range=range(0, 10),
    #     data_path=data_path,
    #     transform=transform, 
    #     train=True, 
    #     batch_size=64
    # )

    test_dataloader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10), 
        data_path=data_path,
        transform=transform, 
        train=False,
        num_samples=128,
        batch_size=64
    )


    # Model loading
    net = sffnn_batched.Net(
        num_inputs=350,      # Nach Downsample1D(0.5): 700 -> 350
        num_hidden1=128,     # Erstes Hidden Layer
        num_hidden2=64,      # Zweites Hidden Layer (hierarchisch)
        num_outputs=10, 
        num_steps=80,      # 80 Zeitschritte (entspricht n_time_bins)
        beta=0.9
    ).to(device)


    # Training Setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    num_epochs = 10
    
    trainer = Trainer(net, optimizer, loss_fn, device, project_root=project_root)
    
    # Konfiguration für Activity Monitoring
    MONITORING_CONFIG = {
        'num_samples': 64,
        'layer_names': ['lif0', 'lif1', 'lif2', 'lif3'],
        'save_dir': os.path.join(project_root, "data", "activity_logs")
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.evaluate(test_dataloader)
        
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        # Activity Monitoring (nur in bestimmten Epochen)
        print(f"\n🧪 Activity Monitoring nach Epoch {epoch}:")
        metadata_extractor = SHDMetadataExtractor()
        input_transform = lambda x: x.squeeze(2) if x.ndim == 4 else x
        
        activity_monitor = ActivityMonitor(
            net,
            metadata_extractor=metadata_extractor,
            input_transform=input_transform
        )
        activity_monitor.enable_monitoring(lif_layer_names=MONITORING_CONFIG['layer_names'])
        
        activity_monitor.monitor_and_save_samples(
            dataloader=test_dataloader,
            num_samples=MONITORING_CONFIG['num_samples'],
            layer_names=MONITORING_CONFIG['layer_names'],
            save_dir=MONITORING_CONFIG['save_dir'],
            epoch=epoch,
            device=device,
            verbose=True
        )
        
        activity_monitor.disable_monitoring()
    
    # Speichere Performance-Plots
    plot_path = trainer.save_plots()
    print(f"\n✅ Performance-Plots gespeichert: {plot_path}")

    # Save the model
    model_export_path = os.path.join(project_root, "models", "model_export")
    os.makedirs(model_export_path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(model_export_path, "model_weights.pth"))
    print(f"\n✅ Model weights saved: {os.path.join(model_export_path, 'model_weights.pth')}")  

    


    ## Qualitative Analysis of Manifold Visualization of UMAP,PCA, tSNE,Isomap,MDS, SpectralEmbedding, LocallyLinearEmbedding

import torch.nn as nn
import os
import json
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
from manifolduntanglinganalysis.training import Trainer
import models.sffnn_batched as sffnn_batched
from manifolduntanglinganalysis.ActivityMonitor import ActivityMonitor
import manifolduntanglinganalysis.analysis.intrinsic_dimension as id_analysis
#from manifolduntanglinganalysis.analysis.intrinsic_dimension import plot_intrinsic_dimensions_over_layers
from manifolduntanglinganalysis.preprocessing.metadata_extractor import SHDMetadataExtractor
from manifolduntanglinganalysis.metrics.mean_field_theoretic_manifold_analysis_wrapper import analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds, plot_manifold_metrics_over_epochs, analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded, plot_manifold_metrics_over_layer, plot_manifold_metrics_over_epochs_all_layer_in_one_plot
import numpy as np
import random
import torch
import h5py
import re
from tonic.transforms import ToFrame
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
# Sortiere Activity Logs nach Epoche und Layer
# Format: epoch_XXX_layername_spk_events.h5
def sort_key(filename):
    match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', filename)
    if match:
        epoch = int(match.group(1))
        layer = match.group(2)
        # Sortiere zuerst nach Epoche, dann nach Layer
        return (epoch, layer)
    return (999, 'zzz')  # Unbekannte Dateien ans Ende

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
    train_dataloader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10),
        data_path=data_path,
        transform=transform, 
        train=True, 
        batch_size=64
    )

    test_dataloader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 10), 
        data_path=data_path,
        transform=transform, 
        train=False,
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
        'num_samples': 1000,
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
    
    # Speichere Performance-Metriken in JSON
    results_dir = os.path.join(project_root, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Konvertiere Trainer-History zu einem strukturierten Format
    performance_metrics = {
        'epochs': list(range(1, len(trainer.history['train_loss']) + 1)),
        'train_loss': [float(x) for x in trainer.history['train_loss']],
        'train_accuracy': [float(x) for x in trainer.history['train_accuracy']],
        'val_loss': [float(x) for x in trainer.history['val_loss']],
        'val_accuracy': [float(x) for x in trainer.history['val_accuracy']],
        'val_precision': [float(x) for x in trainer.history['val_precision']],
        'val_recall': [float(x) for x in trainer.history['val_recall']],
        'val_f1': [float(x) for x in trainer.history['val_f1']],
        'val_auc_roc': [float(x) for x in trainer.history['val_auc_roc']]
    }
    
    performance_json_path = os.path.join(results_dir, "performance_metrics.json")
    with open(performance_json_path, 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    print(f"✅ Performance-Metriken gespeichert: {performance_json_path}")

    # Save the model
    model_export_path = os.path.join(project_root, "models", "model_export")
    os.makedirs(model_export_path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(model_export_path, "model_weights.pth"))
    print(f"\n✅ Model weights saved: {os.path.join(model_export_path, 'model_weights.pth')}")  

    

    results_input = {
        'capacity': 0.0069,
        'radius': 1.8510,
        'dimension': 187.0195,
        # Ich füge die anderen Werte aus deinem Text auch hinzu, falls du sie brauchst:
        'correlation': 0.5949,
        'optimal_k': 2
    }

    
    # results_input = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
    #     dataloader=test_dataloader,
    #     labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     max_samples_per_class=100,
    #     kappa=0.0,
    #     n_t=200,
    #     n_reps=1,
    #     verbose=True
    # )
    print(f"Capacity: {results_input['capacity']:.4f}")
    print(f"Radius: {results_input['radius']:.4f}")
    print(f"Dimension: {results_input['dimension']:.4f}")
    #construct path of all activity logs
    activity_logs_path = os.path.join(project_root, "data", "activity_logs")
    activity_logs = os.listdir(activity_logs_path)
    

    
    activity_logs = sorted(activity_logs, key=sort_key)
    
    results = {}  # Struktur: results[epoch][layer] = {'capacity': ..., 'radius': ..., 'dimension': ...}
    results_list = []  # Für plot_manifold_metrics_over_epochs
    pca_intdims = []
    for activity_log in activity_logs:
        # Parse Epoch und Layer aus dem Dateinamen
        match = re.match(r'epoch_(\d+)_(\w+)_spk_events\.h5', activity_log)
        if not match:
            print(f"Warnung: Konnte Epoch und Layer nicht aus {activity_log} extrahieren")
            continue
        
        epoch = int(match.group(1))
        layer = match.group(2)
        
        # Lade Activity Log und erstelle Transform mit korrekter sensor_size
        activity_log_path = os.path.join(project_root, "data", "activity_logs", activity_log)
        
        with h5py.File(activity_log_path, 'r') as f:
            num_neurons = int(f.attrs['num_features']) 
        
        # Erstelle Transform mit korrekter sensor_size (Format: (neurons, height, width))
        activity_log_transform = datatransforms.get_activity_logpreprocessing(num_neurons=num_neurons,fixed_duration=80,n_time_bins=10)
        
        # Lade Activity Log mit Transform
        activity_log_dataloader = dataloader.load_activity_log(
            activity_log_path=activity_log_path, 
            transform=activity_log_transform
        )
        print(f"Analyzing activity log: {activity_log}")
        current_result = None
        # if "lif3" in activity_log:
        #     current_result = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded(
        #         dataloader=activity_log_dataloader,
        #         labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #         max_samples_per_class=100,
        #         kappa=0.0,
        #         n_t=200,
        #         n_reps=1,
        #         verbose=True
        #     )
        # else:
        current_result = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
            dataloader=activity_log_dataloader,
            labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            max_samples_per_class=64,
            kappa=0.0,
            n_t=200,
            n_reps=1,
            verbose=True
        )
        #intrinsic dimension calculation with PCA 80% explained vairance and as comparison MLE
        pca_intdim,fig=id_analysis.explained_variance_dimension(activity_log_dataloader,perc=0.90,plot_path=os.path.join(project_root,"plots","explained_variance_dimension.png"))
        #mle_dim = id_analysis.mle_intrinsic_dimension(activity_log_dataloader)
        print(f"PCA Intrinsic Dimension: {pca_intdim}")
       # print(f"MLE Intrinsic Dimension: {mle_dim}")


        # 2. Auf die temporäre Variable zugreifen für den Print
        print(f"Capacity: {current_result['capacity']:.4f}")
        print(f"Radius: {current_result['radius']:.4f}")
        print(f"Dimension: {current_result['dimension']:.4f}")

        # 3. Strukturiere nach Epoch und Layer
        if epoch not in results:
            results[epoch] = {}
        
        # Konvertiere NumPy-Datentypen zu nativen Python-Typen
        results[epoch][layer] = {
            'capacity': float(current_result['capacity']),
            'radius': float(current_result['radius']),
            'dimension': float(current_result['dimension']),
            'correlation': float(current_result['correlation']),
            'pca_intdim': float(pca_intdim),
            #'mle_dim': float(mle_dim)
        }
        
        # Für plot_manifold_metrics_over_epochs die vollständigen Ergebnisse behalten
        results_list.append(current_result)
    
    # Speichere die Ergebnisse in JSON-Dateien
    import json
    results_dir = os.path.join(project_root, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Speichere results_all.json (wird für Plots benötigt)
    results_json_path = os.path.join(results_dir, "results_all.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Speichere auch results.json (für Kompatibilität)
    with open(os.path.join(results_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Erstelle Plots aus den berechneten Ergebnissen
    plot_manifold_metrics_over_epochs(results_json_path=results_json_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_epochs(results_list, activity_logs, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    plot_manifold_metrics_over_layer(results_json_path=results_json_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_layer(results_list, activity_logs, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    plot_manifold_metrics_over_epochs_all_layer_in_one_plot(results_json_path=results_json_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_epochs_all_layer_in_one_plot(results_list, activity_logs, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    
    # Alternative: Erstelle Plots direkt aus JSON-Datei (kommentiert aus)
    # results_json_path = os.path.join(project_root, "data", "results", "results_all.json")
    # plot_manifold_metrics_over_epochs(results_json_path=results_json_path, input_data_metrics=results_input, save_dir=os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    # plot_manifold_metrics_over_layer(results_json_path=results_json_path, input_data_metrics=results_input, save_dir=os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    # plot_manifold_metrics_over_epochs_all_layer_in_one_plot(results_json_path=results_json_path, input_data_metrics=results_input, save_dir=os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))

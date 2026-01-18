import torch.nn as nn
import os
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
from manifolduntanglinganalysis.training import Trainer
import models.sffnn_batched as sffnn_batched
import models.snn_r_leaky as snn_r_leaky
from manifolduntanglinganalysis.ActivityMonitor import ActivityMonitor
from manifolduntanglinganalysis.preprocessing.metadata_extractor import SHDMetadataExtractor
from manifolduntanglinganalysis.metrics.mean_field_theoretic_manifold_analysis_wrapper import analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds, plot_manifold_metrics_over_epochs, analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds_rate_coded, plot_manifold_metrics_over_layer, plot_manifold_metrics_over_epochs_all_layer_in_one_plot
import numpy as np
import random
import torch
import h5py
import re
from tonic.transforms import ToFrame
import manifolduntanglinganalysis.analysis.intrinsic_dimension as id_analysis
import os
import models.snn_r_leaky as snn_r_leaky


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
        n_time_bins=500,
        target_neurons=350,
        original_neurons=700,
        fixed_duration=958007.0
    )

    # Data loading
    train_dataloader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 20),
        data_path=data_path,
        transform=transform, 
        train=True, 
        batch_size=64
    )

    test_dataloader = dataloader.load_filtered_shd_dataloader(
        label_range=range(0, 20), 
        data_path=data_path,
        transform=transform, 
        train=False,
        batch_size=64
    )


    # Model loading - Rekurrentes SNN
    net = snn_r_leaky.RSNN(
        num_inputs=350,      # Nach Downsample1D(0.5): 700 -> 350
        num_hidden=256,      # Größe des rekurrenten Hidden Layers
        num_outputs=20, 
        num_steps=500,        # 80 Zeitschritte (entspricht n_time_bins)
        beta=0.5
    ).to(device)


    # Training Setup
    loss_fn = nn.CrossEntropyLoss()
    # Sehr niedrige Learning Rate für rekurrente Netze (verhindert Instabilität)
    # Rekurrente Netze benötigen oft sehr kleine Learning Rates
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)  # Reduziert auf 5e-5 für Stabilität
    

    num_epochs = 100
    
    trainer = Trainer(net, optimizer, loss_fn, device, project_root=project_root)
    
    # Konfiguration für Activity Monitoring
    # Für rekurrentes Modell: rlif (recurrent hidden) und lif_output (output)
    MONITORING_CONFIG = {
        'num_samples': 1000,
        'layer_names': ['rlif', 'lif_output'],  # Rekurrentes Hidden Layer und Output Layer
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

    

    # results_input = {
    #     'capacity': 0.0069,
    #     'radius': 1.8510,
    #     'dimension': 187.0195,
    #     # Ich füge die anderen Werte aus deinem Text auch hinzu, falls du sie brauchst:
    #     'correlation': 0.5949,
    #     'optimal_k': 2
    # }

    global_dimension_input = {
        'PCA' : 27,
        'MLE' : 25.95,
        'Two-NN' : 31.02
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
    # print(f"Capacity: {results_input['capacity']:.4f}")
    # print(f"Radius: {results_input['radius']:.4f}")
    # print(f"Dimension: {results_input['dimension']:.4f}")
    #construct path of all activity logs
    activity_logs_path = os.path.join(project_root, "data", "activity_logs_feed_forward")
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
        activity_log_transform = datatransforms.get_activity_logpreprocessing(num_neurons=num_neurons,fixed_duration=80,n_time_bins=80)
        
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
        # current_result = analyze_manifold_capacity_and_mftma_metrics_of_class_manifolds(
        #     dataloader=activity_log_dataloader,
        #     labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     max_samples_per_class=100,
        #     kappa=0.0,
        #     n_t=200,
        #     n_reps=1,
        #     verbose=True
        # )

        # # 2. Auf die temporäre Variable zugreifen für den Print
        # print(f"Capacity: {current_result['capacity']:.4f}")
        # print(f"Radius: {current_result['radius']:.4f}")
        # print(f"Dimension: {current_result['dimension']:.4f}")
        pca_intdim,fig=id_analysis.explained_variance_dimension(activity_log_dataloader,perc=0.80,plot_path=os.path.join(project_root,"plots","explained_variance_dimension.png"))
        mle_dim = id_analysis.mle_intrinsic_dimension(activity_log_dataloader)
        twonn_dim = id_analysis.twonn_intrinsic_dimension(activity_log_dataloader)
        current_result = {
            'PCA':pca_intdim,
            'MLE':mle_dim,
            'Two-NN':twonn_dim
        }
        # 3. Strukturiere nach Epoch und Layer
        if epoch not in results:
            results[epoch] = {}
    #   results[epoch][layer] = {
    #     'capacity': float(current_result['capacity']),
    #     'radius': float(current_result['radius']),
    #     'dimension': float(current_result['dimension']),
    #     'correlation': float(current_result['correlation'])
    # }
    

        # Konvertiere NumPy-Datentypen zu nativen Python-Typen
        results[epoch][layer] = {
            'PCA': float(current_result['PCA']),
            'MLE': float(current_result['MLE']),
            'Two-NN': float(current_result['Two-NN'])
        }
        
        # Für plot_manifold_metrics_over_epochs die vollständigen Ergebnisse behalten
        results_list.append(current_result)
    
    # Erstelle Plots aus den berechneten Ergebnissen
    results_json_path = os.path.join(project_root, "data", "results", "results_all.json")
    #plot_manifold_metrics_over_epochs(results_json_path=results_json_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_epochs(results_list, activity_logs, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_layer(results_json_path=results_json_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_layer(results_list, activity_logs, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
   #plot_manifold_metrics_over_epochs_all_layer_in_one_plot(results_json_path=results_json_path, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    #plot_manifold_metrics_over_epochs_all_layer_in_one_plot(results_list, activity_logs, input_data_metrics=results_input, save_dir= os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    
    # Alternative: Erstelle Plots direkt aus JSON-Datei (kommentiert aus)
    # results_json_path = os.path.join(project_root, "data", "results", "results_all.json")
    # plot_manifold_metrics_over_epochs(results_json_path=results_json_path, input_data_metrics=results_input, save_dir=os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    # plot_manifold_metrics_over_layer(results_json_path=results_json_path, input_data_metrics=results_input, save_dir=os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    # plot_manifold_metrics_over_epochs_all_layer_in_one_plot(results_json_path=results_json_path, input_data_metrics=results_input, save_dir=os.path.join(project_root, "plots"), figsize_per_subplot=(5, 4))
    # save the results in a json file


    id_analysis.plot_intrinsic_dimensions_over_layers(
    results=results,  # {epoch: {layer: {'PCA': float, 'MLE': float, 'Two-NN': float}}}
    input_data_metrics=None,  # Optional: {'PCA': float, 'MLE': float, 'Two-NN': float}
    save_dir=os.path.join(project_root, "plots"),
    figsize_per_subplot=(6, 4)
)
    import json
    # Stelle sicher, dass das Verzeichnis existiert
    results_dir = os.path.join(project_root, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results_GLOB_DIM.json"), 'w') as f:
        json.dump(results, f, indent=2)

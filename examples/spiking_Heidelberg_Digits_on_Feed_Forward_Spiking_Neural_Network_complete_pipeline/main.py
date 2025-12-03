import torch.nn as nn
import os
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
from manifolduntanglinganalysis.training import Trainer
import models.sffnn_batched as sffnn_batched
from manifolduntanglinganalysis.ActivityMonitor import ActivityMonitor
from manifolduntanglinganalysis.preprocessing.metadata_extractor import SHDMetadataExtractor
import numpy as np
import random
import torch
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

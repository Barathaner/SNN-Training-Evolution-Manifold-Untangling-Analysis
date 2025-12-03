import torch.nn as nn
import os
import manifolduntanglinganalysis.preprocessing.datatransforms as datatransforms
import manifolduntanglinganalysis.preprocessing.dataloader as dataloader
import models.trainer as trainer
import models.sffnn_batched as sffnn_batched
import manifolduntanglinganalysis.metrics.performance_metrics as metrics
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


    # Training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    num_epochs = 20
    
    # Konfiguration für Activity Monitoring
    MONITORING_CONFIG = {
        'num_samples': 64,  # Anzahl der Samples, die überwacht werden sollen
        'layer_names': ['lif0', 'lif1', 'lif2', 'lif3'],  # Zu überwachende Layer
        'save_dir': os.path.join(project_root, "data", "activity_logs")
    }
    
   
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Training (OHNE Monitoring - wir wollen nur Test-Aktivitäten)
        trainer.train_one_epoch_batched(
            net=net,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
        
        # Test-Evaluation MIT Monitoring
        print(f"\n🧪 Test-Evaluation nach Epoch {epoch}:")
        net.eval()
        
        # Activity Monitoring: Überwache und speichere Spikes
        # Erstelle MetadataExtractor für SHD
        metadata_extractor = SHDMetadataExtractor()
        # Input-Transformation für SHD-Format (optional, da als Fallback vorhanden)
        input_transform = lambda x: x.squeeze(2) if x.ndim == 4 else x
        
        activity_monitor = ActivityMonitor(
            net,
            metadata_extractor=metadata_extractor,
            input_transform=input_transform
        )
        activity_monitor.enable_monitoring(lif_layer_names=MONITORING_CONFIG['layer_names'])
        
        filepaths = activity_monitor.monitor_and_save_samples(
            dataloader=test_dataloader,
            num_samples=MONITORING_CONFIG['num_samples'],
            layer_names=MONITORING_CONFIG['layer_names'],
            save_dir=MONITORING_CONFIG['save_dir'],
            epoch=epoch,
            device=device,
            verbose=True
        )
        
        # Monitoring deaktivieren
        activity_monitor.disable_monitoring()

        ## 
        accuracy = metrics.print_full_dataloader_accuracy_batched(net, test_dataloader)
        print(f"\n   Test Accuracy: {accuracy:.4f}")
        
        net.train()  # Zurück zum Trainingsmodus
    
        
    
    # Testing
    print(f"\n{'='*80}")
    print("Finale Evaluation")
    print(f"{'='*80}")
    metrics.print_full_dataloader_accuracy_batched(net, test_dataloader)

    # Save the model
    model_export_path = os.path.join(project_root, "models", "model_export")
    os.makedirs(model_export_path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(model_export_path, "model_weights.pth"))
    print(f"\n✅ Model weights saved: {os.path.join(model_export_path, 'model_weights.pth')}")  
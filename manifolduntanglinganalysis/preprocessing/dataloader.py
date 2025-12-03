import tonic
from tonic.transforms import ToFrame
from tonic import DiskCachedDataset
from tonic.collation import PadTensors
from torch.utils.data import Subset, Dataset, DataLoader

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]
        return self.transform(events), label

def load_filtered_shd_dataloader(
    label_range=range(0, 10),
    data_path="./data",
    transform=None,
    train=True,
    batch_size=32,
    shuffle=False,
    drop_last=True,
    num_workers=0,
):
    """
    Lädt einen gefilterten und transformierten SHD-Datensatz als DataLoader.

    Args:
        label_range (iterable): Liste oder Range der gewünschten Labels (default: 0–9)
        transform (callable): Event → Tensor Transform (default: ToFrame)
        train (bool): Trainings- oder Testset
        batch_size (int): Batchgröße für DataLoader
        shuffle (bool): Zufällige Batchmischung
        drop_last (bool): Letzten Batch verwerfen, wenn zu klein
        num_workers (int): DataLoader Worker

    Returns:
        torch.utils.data.DataLoader: Dataloader für das vorbereitete SHD-Dataset
    """
    dataset_full = tonic.datasets.SHD(save_to=data_path, train=train, transform=None)

    label_range = set(label_range)
    filtered_indices = [
        i for i in range(len(dataset_full)) if dataset_full[i][1] in label_range
    ]

    if transform is None:
        transform = ToFrame(
            sensor_size=tonic.datasets.SHD.sensor_size,  # = (700,)
            n_time_bins=1000  # Aktualisiert auf 1000
        )
    subset = Subset(dataset_full, filtered_indices)
    transformed_dataset = TransformedDataset(subset, transform)

    # Cache-Pfad basierend auf n_time_bins, damit unterschiedliche Konfigurationen nicht kollidieren
    cache_suffix = f"n_time_bins_{transform.n_time_bins if hasattr(transform, 'n_time_bins') else 'default'}"
    cached_dataset = DiskCachedDataset(transformed_dataset, cache_path=f'./cache/shd/{cache_suffix}')
    dataloader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=PadTensors(),
        num_workers=num_workers,
    )

    return dataloader

def load_shd_raw_subset(train=False, label_range=range(0, 9)):
    dataset_full = tonic.datasets.SHD(save_to="./data", train=train, transform=None)

    label_range = set(label_range)
    filtered_indices = [
        i for i in range(len(dataset_full)) if dataset_full[i][1] in label_range
    ]
    
    subset = Subset(dataset_full, filtered_indices)
    return subset
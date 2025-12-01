import torch
def print_full_dataloader_accuracy(net,dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    correct = 0
    total = 0

    net.eval()  # Setzt das Netz in den Evaluationsmodus

    with torch.no_grad():
        for events, labels in dataloader:
            events = events.to(device).float()  # Wichtig: float!
            labels = labels.to(device)
            batch_size = events.shape[0]
            for i in range(batch_size):
                sample = events[i].squeeze(0)  # [250, 700]
                spk_rec, mem_rec = net(sample)
                spike_sums = spk_rec.sum(dim=0)
                predicted_label = torch.argmax(spike_sums).item()
                if predicted_label == labels[i].item():
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")


def print_batch_accuracy(net, batch):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    correct = 0
    total = 0

    net.eval()  # Setzt das Netz in den Evaluationsmodus

    with torch.no_grad():
        events, labels = batch
        events = events.to(device).float()
        labels = labels.to(device)

        batch_size = events.shape[0]
        for i in range(batch_size):
            sample = events[i].squeeze(0)  # z. B. [250, 700]
            spk_rec, mem_rec = net(sample)
            spike_sums = spk_rec.sum(dim=0)
            predicted_label = torch.argmax(spike_sums).item()
            if predicted_label == labels[i].item():
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Batch Accuracy: {accuracy:.4f}")
    
def print_full_dataloader_accuracy_batched(net, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for events, labels in dataloader:
            # Formatanpassung: [B, T, 1, 700] → [B, T, 700]
            if events.ndim == 4:
                events = events.squeeze(2)
            events = events.to(device).float()
            labels = labels.to(device)

            spk_rec, _ = net(events)  # [B, T, num_outputs]
            spike_sums = spk_rec.sum(dim=1)  # [B, num_outputs]
            preds = torch.argmax(spike_sums, dim=1)  # [B]

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy (Full Dataloader): {accuracy:.4f}")
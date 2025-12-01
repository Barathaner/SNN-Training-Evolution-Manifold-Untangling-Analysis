import torch
def train_one_epoch_with_rate_encoding_for_sample(net, dataloader, optimizer, loss_fn, device):
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for events, labels in dataloader:
        events = events.to(device).float()
        labels = labels.to(device)

        batch_size = events.shape[0]

        for i in range(batch_size):
            sample = events[i].squeeze(0)  # [num_steps, 700]

            optimizer.zero_grad()

            spk_rec, mem_rec = net(sample)

            # Rate Coding: Summe aller Spikes als Logits
            spike_sums = spk_rec.sum(dim=0)

            loss = loss_fn(spike_sums, labels[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            pred = torch.argmax(spike_sums).item()
            if pred == labels[i].item():
                correct += 1
            total += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def train_one_epoch_batched(net, dataloader, optimizer, loss_fn, device):
    """
    Training für ein batchfähiges SNN-Modell mit Rate-Coding (Summe über Zeit).
    
    Args:
        net: ein batchfähiges snnTorch-Modell
        dataloader: torch DataLoader mit shape [B, T, 1, 700] oder [B, T, 700]
        optimizer: z. B. Adam
        loss_fn: z. B. nn.CrossEntropyLoss()
        device: torch.device("cuda") etc.
    """
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for events, labels in dataloader:
        # Events von [B, T, 1, 700] → [B, T, 700]
        if events.ndim == 4:
            events = events.squeeze(2)
        
        events = events.to(device).float()  # wichtig für snnTorch
        labels = labels.to(device)

        optimizer.zero_grad()
        spk_rec, _ = net(events)  # spk_rec: [B, T, num_outputs]

        # Rate Coding: Zeitliche Summe der Spikes pro Output-Neuron
        spike_sums = spk_rec.sum(dim=1)  # → [B, num_outputs]

        loss = loss_fn(spike_sums, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)  # batch loss
        preds = torch.argmax(spike_sums, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

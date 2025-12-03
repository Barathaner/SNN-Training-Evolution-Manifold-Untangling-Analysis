import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, project_root: Optional[str] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc_roc': []
        }
    
    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for events, labels in dataloader:
            if events.ndim == 4:
                events = events.squeeze(2)
            
            events = events.to(self.device).float()
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            spk_rec, _ = self.model(events)
            spike_sums = spk_rec.sum(dim=1)
            
            loss = self.loss_fn(spike_sums, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(spike_sums, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        total_samples = len(all_labels)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = accuracy_score(all_labels, all_preds)
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for events, labels in dataloader:
                if events.ndim == 4:
                    events = events.squeeze(2)
                
                events = events.to(self.device).float()
                labels = labels.to(self.device)
                
                spk_rec, _ = self.model(events)
                spike_sums = spk_rec.sum(dim=1)
                
                loss = self.loss_fn(spike_sums, labels)
                total_loss += loss.item() * labels.size(0)
                
                probs = torch.softmax(spike_sums, dim=1)
                preds = torch.argmax(spike_sums, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        total_samples = len(all_labels)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        try:
            if len(np.unique(all_labels)) > 1:
                auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
            else:
                auc_roc = 0.0
        except:
            auc_roc = 0.0
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_accuracy'].append(accuracy)
        self.history['val_precision'].append(precision)
        self.history['val_recall'].append(recall)
        self.history['val_f1'].append(f1)
        self.history['val_auc_roc'].append(auc_roc)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def save_plots(self, save_path: Optional[str] = None):
        if save_path is None:
            plots_dir = self.project_root / 'plots'
            plots_dir.mkdir(exist_ok=True)
            save_path = plots_dir / 'PerformanceMetriken.png'
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs, self.history['val_precision'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision (Validation)')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.history['val_recall'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall / Sensitivity (Validation)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.history['val_f1'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('F1-Score (Validation)')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(epochs, self.history['val_auc_roc'], 'orange', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC-ROC')
        axes[1, 2].set_title('AUC-ROC (Validation)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


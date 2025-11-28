"""
Federated Learning for Medical Imaging GANs
Implements FedAvg algorithm for training GANs across distributed clients
"""

import os
import glob
import pickle
import copy
import yaml
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import spectral_norm as SN
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


# Dataset Classes

class PKLPairDataset(Dataset):
    """Dataset for loading preprocessed chest X-ray images from PKL files"""
    def __init__(self, root, labels=["NORMAL", "PNEUMONIA"]):
        self.samples = []
        for label_name in labels:
            for p in glob.glob(str(Path(root) / "train" / label_name / "*.pkl")):
                self.samples.append((p, 0 if label_name == "NORMAL" else 1))
        if not self.samples:
            raise RuntimeError(f"No PKL files found under train/{labels}/")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        with open(p, "rb") as f:
            obj = pickle.load(f)
        x = torch.from_numpy(obj["image"]).float() / 127.5 - 1.0
        x = x.unsqueeze(0)
        return x, y

# Model Architectures (cGAN)

class Generator(nn.Module):
    """Conditional GAN Generator with class embedding"""
    def __init__(self, z_dim=128, num_classes=2, embed_dim=50, base_ch=512, out_ch=1):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        in_dim = z_dim + embed_dim
        self.fc = nn.Linear(in_dim, base_ch * 4 * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch//2, 4, 2, 1),
            nn.BatchNorm2d(base_ch//2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch//2, base_ch//4, 4, 2, 1),
            nn.BatchNorm2d(base_ch//4),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch//4, base_ch//8, 4, 2, 1),
            nn.BatchNorm2d(base_ch//8),
            nn.ReLU(True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_ch//8, base_ch//16, 4, 2, 1),
            nn.BatchNorm2d(base_ch//16),
            nn.ReLU(True)
        )
        self.to_img = nn.ConvTranspose2d(base_ch//16, out_ch, 4, 2, 1)
    
    def forward(self, z, y):
        e = self.embed(y)
        h = torch.cat([z, e], dim=1)
        h = self.fc(h).view(h.size(0), -1, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return torch.tanh(self.to_img(h))


class Discriminator(nn.Module):
    """Conditional GAN Discriminator with spectral normalization"""
    def __init__(self, num_classes=2, in_ch=1, base_ch=64):
        super().__init__()
        self.c1 = SN(nn.Conv2d(in_ch, base_ch, 4, 2, 1))
        self.c2 = SN(nn.Conv2d(base_ch, base_ch*2, 4, 2, 1))
        self.c3 = SN(nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1))
        self.c4 = SN(nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1))
        self.c5 = SN(nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1))
        self.lin = SN(nn.Linear(base_ch*8, 1))
        self.embed = SN(nn.Embedding(num_classes, base_ch*8))
    
    def forward(self, x, y):
        h = F.leaky_relu(self.c1(x), 0.2, inplace=True)
        h = F.leaky_relu(self.c2(h), 0.2, inplace=True)
        h = F.leaky_relu(self.c3(h), 0.2, inplace=True)
        h = F.leaky_relu(self.c4(h), 0.2, inplace=True)
        h = F.leaky_relu(self.c5(h), 0.2, inplace=True)
        h = h.sum(dim=[2, 3])
        out = self.lin(h).squeeze(1)
        out += torch.sum(self.embed(y) * h, dim=1)
        return out


# ============================================================================
# Loss Functions
# ============================================================================

def d_hinge_loss(real_logits, fake_logits):
    """Hinge loss for discriminator"""
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    """Hinge loss for generator"""
    return -fake_logits.mean()


# Federated Learning Components

class FederatedServer:
    """
    Federated Learning Server
    Manages global model and aggregates updates from clients
    """
    def __init__(self, model_G, model_D, device):
        self.global_G = model_G.to(device)
        self.global_D = model_D.to(device)
        self.device = device
        
    def get_global_models(self):
        """Return copies of current global models"""
        return (
            copy.deepcopy(self.global_G.state_dict()),
            copy.deepcopy(self.global_D.state_dict())
        )
    
    def aggregate_models(self, client_models, client_weights=None):
        """
        Aggregate client models using FedAvg
        
        Args:
            client_models: List of (G_state_dict, D_state_dict, num_samples) tuples
            client_weights: Optional custom weights for each client
        """
        if not client_models:
            return
        
        # Calculate weights based on number of samples if not provided
        if client_weights is None:
            total_samples = sum(num_samples for _, _, num_samples in client_models)
            client_weights = [num_samples / total_samples for _, _, num_samples in client_models]
        
        # Aggregate Generator
        global_G_state = OrderedDict()
        for key in self.global_G.state_dict().keys():
            global_G_state[key] = sum(
                client_G[key] * weight
                for (client_G, _, _), weight in zip(client_models, client_weights)
            )
        self.global_G.load_state_dict(global_G_state)
        
        # Aggregate Discriminator
        global_D_state = OrderedDict()
        for key in self.global_D.state_dict().keys():
            global_D_state[key] = sum(
                client_D[key] * weight
                for (_, client_D, _), weight in zip(client_models, client_weights)
            )
        self.global_D.load_state_dict(global_D_state)
    
    def save_global_models(self, save_dir, round_num):
        """Save global models to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save(
            self.global_G.state_dict(),
            save_dir / f"federated_G_round_{round_num}.pth"
        )
        torch.save(
            self.global_D.state_dict(),
            save_dir / f"federated_D_round_{round_num}.pth"
        )


class FederatedClient:
    """
    Federated Learning Client
    Trains local models on private data
    """
    def __init__(self, client_id, dataloader, device, z_dim=128):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.z_dim = z_dim
        self.local_G = None
        self.local_D = None
        self.g_opt = None
        self.d_opt = None
        
    def setup_models(self, global_G_state, global_D_state, lr_g=2e-4, lr_d=2e-4):
        """Initialize local models with global state"""
        self.local_G = Generator(z_dim=self.z_dim).to(self.device)
        self.local_D = Discriminator().to(self.device)
        
        self.local_G.load_state_dict(global_G_state)
        self.local_D.load_state_dict(global_D_state)
        
        self.g_opt = torch.optim.Adam(
            self.local_G.parameters(),
            lr=lr_g,
            betas=(0.0, 0.9)
        )
        self.d_opt = torch.optim.Adam(
            self.local_D.parameters(),
            lr=lr_d,
            betas=(0.0, 0.9)
        )
    
    def train_local_epochs(self, num_epochs, verbose=False):
        """Train local models for specified number of epochs"""
        self.local_G.train()
        self.local_D.train()
        
        total_samples = 0
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            if verbose:
                pbar = tqdm(
                    self.dataloader,
                    desc=f"Client {self.client_id} - Epoch {epoch+1}/{num_epochs}",
                    ncols=100
                )
            else:
                pbar = self.dataloader
            
            g_losses, d_losses = [], []
            
            for real, y in pbar:
                real = real.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_size = real.size(0)
                total_samples += batch_size
                
                # Train Discriminator
                self.d_opt.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                with torch.no_grad():
                    fake = self.local_G(z, y)
                
                real_logits = self.local_D(real, y)
                fake_logits = self.local_D(fake.detach(), y)
                d_loss = d_hinge_loss(real_logits, fake_logits)
                d_loss.backward()
                self.d_opt.step()
                d_losses.append(d_loss.item())
                
                # Train Generator
                self.g_opt.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake = self.local_G(z, y)
                fake_logits = self.local_D(fake, y)
                g_loss = g_hinge_loss(fake_logits)
                g_loss.backward()
                self.g_opt.step()
                g_losses.append(g_loss.item())
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        "d_loss": f"{d_loss.item():.3f}",
                        "g_loss": f"{g_loss.item():.3f}"
                    })
            
            epoch_metrics.append({
                'g_loss': np.mean(g_losses),
                'd_loss': np.mean(d_losses)
            })
        
        return (
            self.local_G.state_dict(),
            self.local_D.state_dict(),
            len(self.dataloader.dataset),
            epoch_metrics
        )


# ============================================================================
# Data Partitioning for Federated Learning
# ============================================================================

def partition_dataset(dataset, num_clients, partition_type='iid', alpha=0.5):
    """
    Partition dataset across clients
    
    Args:
        dataset: Full dataset
        num_clients: Number of clients
        partition_type: 'iid' or 'non_iid'
        alpha: Dirichlet parameter for non-IID partitioning (lower = more skewed)
    
    Returns:
        List of indices for each client
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    if partition_type == 'iid':
        # IID: Random uniform partitioning
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
        return [idx.tolist() for idx in client_indices]
    
    elif partition_type == 'non_iid':
        # Non-IID: Use Dirichlet distribution for label skew
        labels = np.array([dataset[i][1] for i in indices])
        num_classes = len(np.unique(labels))
        
        client_indices = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Split indices according to proportions
            splits = np.split(idx_k, proportions)
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())
        
        # Shuffle within each client
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
        
        return client_indices
    
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


# Main Federated Learning Function

def federated_learning(
    config_path="config.yaml",
    num_clients=5,
    num_rounds=10,
    local_epochs=3,
    batch_size=32,
    partition_type='iid',
    alpha=0.5,
    save_dir="models/federated",
    seed=42
):
    """
    Main federated learning training loop
    
    Args:
        config_path: Path to config.yaml
        num_clients: Number of federated clients (simulated hospitals)
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per round
        batch_size: Batch size for training
        partition_type: 'iid' or 'non_iid'
        alpha: Dirichlet parameter for non-IID (lower = more skewed)
        save_dir: Directory to save models and results
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cfg = yaml.safe_load(open(config_path, "r"))
    target_dir = Path(cfg["target_dir"])
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    full_dataset = PKLPairDataset(target_dir)
    print(f"Total samples: {len(full_dataset)}")
    
    print(f"Partitioning dataset: {partition_type} (alpha={alpha if partition_type=='non_iid' else 'N/A'})")
    client_indices = partition_dataset(full_dataset, num_clients, partition_type, alpha)
    
    for i, indices in enumerate(client_indices):
        labels = [full_dataset[idx][1] for idx in indices]
        print(f"  Client {i}: {len(indices)} samples, "
              f"NORMAL: {labels.count(0)}, PNEUMONIA: {labels.count(1)}")
    
    client_loaders = []
    for indices in client_indices:
        subset = Subset(full_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        client_loaders.append(loader)
    
    print("\nInitializing global models...")
    z_dim = 128
    global_G = Generator(z_dim=z_dim)
    global_D = Discriminator()
    
    server = FederatedServer(global_G, global_D, device)
    
    clients = []
    for i, loader in enumerate(client_loaders):
        client = FederatedClient(i, loader, device, z_dim=z_dim)
        clients.append(client)
    
    history = {
        'rounds': [],
        'avg_g_loss': [],
        'avg_d_loss': [],
        'client_metrics': []
    }
    
    print(f"\n{'='*80}")
    print(f"Starting Federated Learning: {num_rounds} rounds, {local_epochs} local epochs")
    print(f"{'='*80}\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*80}")
        
        # Get current global model state
        global_G_state, global_D_state = server.get_global_models()
        
        # Train each client
        client_updates = []
        round_metrics = []
        
        for client in clients:
            print(f"\nTraining Client {client.client_id}...")
            
            # Setup local models with global weights
            client.setup_models(global_G_state, global_D_state)
            
            # Local training
            G_state, D_state, num_samples, metrics = client.train_local_epochs(
                local_epochs,
                verbose=(round_num == 1)  # Show progress bar for first round
            )
            
            client_updates.append((G_state, D_state, num_samples))
            round_metrics.append(metrics)
            
            avg_g = np.mean([m['g_loss'] for m in metrics])
            avg_d = np.mean([m['d_loss'] for m in metrics])
            print(f"  Client {client.client_id}: Avg G Loss: {avg_g:.4f}, Avg D Loss: {avg_d:.4f}")
        
        # Aggregate models on server
        print(f"\nAggregating models from {len(client_updates)} clients...")
        server.aggregate_models(client_updates)
        
        all_g_losses = [m['g_loss'] for client_m in round_metrics for m in client_m]
        all_d_losses = [m['d_loss'] for client_m in round_metrics for m in client_m]
        
        avg_g_loss = np.mean(all_g_losses)
        avg_d_loss = np.mean(all_d_losses)
        
        print(f"\nRound {round_num} Summary:")
        print(f"  Average G Loss: {avg_g_loss:.4f}")
        print(f"  Average D Loss: {avg_d_loss:.4f}")
        
        history['rounds'].append(round_num)
        history['avg_g_loss'].append(avg_g_loss)
        history['avg_d_loss'].append(avg_d_loss)
        history['client_metrics'].append(round_metrics)
        
        # Save models periodically
        if round_num % 2 == 0 or round_num == num_rounds:
            server.save_global_models(save_dir, round_num)
            print(f"  Saved global models to {save_dir}/")
    
    # Save final models
    print(f"\n{'='*80}")
    print("Federated Learning Complete!")
    print(f"{'='*80}\n")
    
    server.save_global_models(save_dir, "final")
    
    plot_federated_training(history, save_dir)
    
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        save_history = {
            'rounds': history['rounds'],
            'avg_g_loss': history['avg_g_loss'],
            'avg_d_loss': history['avg_d_loss']
        }
        json.dump(save_history, f, indent=2)
    
    print(f"Training history saved to {history_path}")
    print(f"Final models saved to {save_dir}/")
    
    return history


def plot_federated_training(history, save_dir):
    """Plot and save training curves"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['rounds'], history['avg_g_loss'], marker='o', label='Generator Loss')
    plt.xlabel('Federated Round')
    plt.ylabel('Loss')
    plt.title('Generator Loss over Federated Rounds')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['rounds'], history['avg_d_loss'], marker='o', color='orange', label='Discriminator Loss')
    plt.xlabel('Federated Round')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss over Federated Rounds')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'federated_training_curves.png', dpi=150)
    plt.close()
    print(f"\nTraining curves saved to {save_dir / 'federated_training_curves.png'}")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning for Medical Imaging GANs"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=5,
        help='Number of federated clients (default: 5)'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=10,
        help='Number of federated rounds (default: 10)'
    )
    parser.add_argument(
        '--local_epochs',
        type=int,
        default=3,
        help='Local training epochs per round (default: 3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--partition',
        type=str,
        choices=['iid', 'non_iid'],
        default='iid',
        help='Data partitioning strategy (default: iid)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Dirichlet alpha for non-IID partitioning (default: 0.5, lower = more skewed)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models/federated',
        help='Directory to save models (default: models/federated)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    federated_learning(
        config_path=args.config,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        partition_type=args.partition,
        alpha=args.alpha,
        save_dir=args.save_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


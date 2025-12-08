"""

Variational Autoencoder (VAE) Baseline for Medical Imaging
Different generative model class - typically more stable but blurrier than GANs
"""

import os
import glob
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class PKLPairDataset(Dataset):
    """Dataset for loading chest X-ray images from PKL files"""
    def __init__(self, root):
        self.samples = []
        for label_name in ["NORMAL", "PNEUMONIA"]:
            for p in glob.glob(str(Path(root) / "train" / label_name / "*.pkl")):
                self.samples.append(p)
        if not self.samples:
            raise RuntimeError("No PKL files found under train/{NORMAL,PNEUMONIA}/")
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        p = self.samples[idx]
        with open(p, "rb") as f:
            obj = pickle.load(f)
        x = torch.from_numpy(obj["image"]).float() / 127.5 - 1.0
        x = x.unsqueeze(0)
        return x


class VAEEncoder(nn.Module):
    """VAE Encoder - maps images to latent distribution parameters"""
    def __init__(self, latent_dim=128, in_ch=1, base_ch=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1),
            nn.BatchNorm2d(base_ch*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1),
            nn.BatchNorm2d(base_ch*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc_mu = nn.Linear(base_ch*8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_ch*8 * 4 * 4, latent_dim)
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder - generates images from latent codes"""
    def __init__(self, latent_dim=128, base_ch=512, out_ch=1):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, base_ch * 4 * 4)
        
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
    
    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return torch.tanh(self.to_img(h))


class VAE(nn.Module):
    """Complete VAE model combining encoder and decoder"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate new samples from the learned distribution"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon_x: Reconstructed images
        x: Original images (in range [-1, 1])
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL term (beta-VAE)
    """
    batch_size = x.size(0)
    
    x_scaled = (x + 1) / 2
    recon_scaled = (recon_x + 1) / 2
    recon_scaled = torch.clamp(recon_scaled, 1e-7, 1 - 1e-7)
    
    recon_loss = F.binary_cross_entropy(recon_scaled, x_scaled, reduction='sum') / batch_size
    
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    return recon_loss + beta * kl_div, recon_loss, kl_div


def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    target_dir = Path(cfg["target_dir"])
    Path("models").mkdir(exist_ok=True)

    ds = PKLPairDataset(target_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    
    print(f"Training VAE on device: {device}")
    print(f"Dataset size: {len(ds)} samples")

    latent_dim = 128
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4, betas=(0.9, 0.999))

    epochs = 125
    beta_start = 0.0  # kl annealing: start with no KL penalty
    beta_end = 0.5    # Gradually increase to this value (lower than 1.0 to prevent collapse)
    
    total_losses = []
    recon_losses = []
    kl_losses = []

    print(f"\nStarting VAE training for {epochs} epochs...")
    print(f"Latent dimension: {latent_dim}")
    print(f"KL annealing: beta {beta_start:.2f} -> {beta_end:.2f} over {epochs} epochs")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        vae.train()
        
        # kl annealing: gradually increase beta
        beta = beta_start + (beta_end - beta_start) * min(1.0, epoch / (epochs * 0.5))
        
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for x in pbar:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            recon_x, mu, logvar = vae(x)
            total_loss, recon_loss, kl_div = vae_loss(recon_x, x, mu, logvar, beta)
            
            # check for nan
            if torch.isnan(total_loss):
                print(f"\nWarning: NaN detected at epoch {epoch}. Skipping batch.")
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_div.item()

            pbar.set_postfix({
                "total": f"{total_loss.item():.3f}",
                "recon": f"{recon_loss.item():.3f}",
                "kl": f"{kl_div.item():.3f}",
                "beta": f"{beta:.3f}"
            })

        total_losses.append(epoch_total_loss / len(dl))
        recon_losses.append(epoch_recon_loss / len(dl))
        kl_losses.append(epoch_kl_loss / len(dl))
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{epochs} (beta={beta:.3f}):")
            print(f"  Total Loss: {total_losses[-1]:.4f}")
            print(f"  Recon Loss: {recon_losses[-1]:.4f}")
            print(f"  KL Loss: {kl_losses[-1]:.4f}")

            vae.eval()
            with torch.no_grad():
                z_test = torch.randn(1, latent_dim, device=device)
                sample = vae.decoder(z_test)
                print(f"  Sample output range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
            vae.train()

    torch.save(vae.state_dict(), 'models/vae_128_125epochs.pth')
    print(f"\nModel saved: vae_128_125epochs.pth")

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs+1), total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Total Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs+1), recon_losses, label='Reconstruction Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs+1), kl_losses, label='KL Divergence', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/vae_training_loss_curve.png')
    plt.close()
    print("Saved VAE checkpoint and loss plot.")


if __name__ == "__main__":
    main()


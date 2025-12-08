"""
Unconditional GAN Baseline for Medical Imaging
No class conditioning - generates generic chest X-rays without class labels
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
from torch.nn.utils import spectral_norm as SN
from tqdm import tqdm

class PKLPairDataset(Dataset):
    """Dataset that loads images but ignores labels for unconditional generation"""
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
        return x  # No label returned


class UnconditionalGenerator(nn.Module):
    """Unconditional GAN Generator - no class conditioning"""
    def __init__(self, z_dim=128, base_ch=512, out_ch=1):
        super().__init__()
        self.fc = nn.Linear(z_dim, base_ch * 4 * 4)
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


class UnconditionalDiscriminator(nn.Module):
    """Unconditional GAN Discriminator - no class conditioning"""
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.c1 = SN(nn.Conv2d(in_ch,     base_ch,   4, 2, 1))
        self.c2 = SN(nn.Conv2d(base_ch,   base_ch*2, 4, 2, 1))
        self.c3 = SN(nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1))
        self.c4 = SN(nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1))
        self.c5 = SN(nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1))
        self.lin = SN(nn.Linear(base_ch*8, 1))
    
    def forward(self, x):
        h = F.leaky_relu(self.c1(x), 0.2, inplace=True)
        h = F.leaky_relu(self.c2(h), 0.2, inplace=True)
        h = F.leaky_relu(self.c3(h), 0.2, inplace=True)
        h = F.leaky_relu(self.c4(h), 0.2, inplace=True)
        h = F.leaky_relu(self.c5(h), 0.2, inplace=True)
        h = h.sum(dim=[2, 3])
        out = self.lin(h).squeeze(1)
        return out


def d_hinge_loss(real_logits, fake_logits):
    """Hinge loss for discriminator"""
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    """Hinge loss for generator"""
    return -fake_logits.mean()


def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    target_dir = Path(cfg["target_dir"])
    Path("models").mkdir(exist_ok=True)

    ds = PKLPairDataset(target_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    
    print(f"Training Unconditional GAN on device: {device}")
    print(f"Dataset size: {len(ds)} samples")

    z_dim = 128
    G = UnconditionalGenerator(z_dim=z_dim).to(device)
    D = UnconditionalDiscriminator().to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    epochs = 125
    g_losses, d_losses = [], []

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}", ncols=90)
        g_epoch_loss, d_epoch_loss = 0, 0

        for real in pbar:
            real = real.to(device, non_blocking=True)
            batch_size = real.size(0)

            # discriminator
            d_opt.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, z_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            real_logits = D(real)
            fake_logits = D(fake.detach())
            d_loss = d_hinge_loss(real_logits, fake_logits)
            d_loss.backward()
            d_opt.step()
            d_epoch_loss += d_loss.item()

            # generator
            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, z_dim, device=device)
            fake = G(z)
            fake_logits = D(fake)
            g_loss = g_hinge_loss(fake_logits)
            g_loss.backward()
            g_opt.step()
            g_epoch_loss += g_loss.item()

            pbar.set_postfix({"d": f"{d_loss.item():.3f}", "g": f"{g_loss.item():.3f}"})

        g_losses.append(g_epoch_loss / len(dl))
        d_losses.append(d_epoch_loss / len(dl))
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{epochs} - Avg D Loss: {d_losses[-1]:.4f}, Avg G Loss: {g_losses[-1]:.4f}")

    torch.save(G.state_dict(), 'models/unconditional_gan_G_128_125epochs.pth')
    torch.save(D.state_dict(), 'models/unconditional_gan_D_128_125epochs.pth')
    print(f"\nModels saved")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, epochs+1), d_losses, label='Discriminator Loss')
    plt.plot(range(1, epochs+1), g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Unconditional GAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/unconditional_gan_training_loss_curve.png')
    plt.close()
    print("Saved unconditional GAN checkpoints and loss plot.")


if __name__ == "__main__":
    main()


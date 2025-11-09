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
    def __init__(self, root):
        self.samples = []
        for label_name in ["NORMAL", "PNEUMONIA"]:
            for p in glob.glob(str(Path(root) / "train" / label_name / "*.pkl")):
                self.samples.append((p, 0 if label_name == "NORMAL" else 1))
        if not self.samples:
            raise RuntimeError("No PKL files found under train/{NORMAL,PNEUMONIA}/")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        with open(p, "rb") as f:
            obj = pickle.load(f)
        x = torch.from_numpy(obj["image"]).float() / 127.5 - 1.0
        x = x.unsqueeze(0)
        return x, y


class Generator(nn.Module):
    def __init__(self, z_dim=128, num_classes=2, embed_dim=50, base_ch=512, out_ch=1):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        in_dim = z_dim + embed_dim
        self.fc = nn.Linear(in_dim, base_ch * 4 * 4)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_ch, base_ch//2, 4, 2, 1), nn.BatchNorm2d(base_ch//2), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_ch//2, base_ch//4, 4, 2, 1), nn.BatchNorm2d(base_ch//4), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base_ch//4, base_ch//8, 4, 2, 1), nn.BatchNorm2d(base_ch//8), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(base_ch//8, base_ch//16,4, 2, 1), nn.BatchNorm2d(base_ch//16),nn.ReLU(True))
        self.to_img = nn.ConvTranspose2d(base_ch//16, out_ch, 4, 2, 1)
    def forward(self, z, y):
        e = self.embed(y)
        h = torch.cat([z, e], dim=1)
        h = self.fc(h).view(h.size(0), -1, 4, 4)
        h = self.up1(h); h = self.up2(h); h = self.up3(h); h = self.up4(h)
        return torch.tanh(self.to_img(h))


class Discriminator(nn.Module):
    def __init__(self, num_classes=2, in_ch=1, base_ch=64):
        super().__init__()
        self.c1 = SN(nn.Conv2d(in_ch,     base_ch,   4, 2, 1))
        self.c2 = SN(nn.Conv2d(base_ch,   base_ch*2, 4, 2, 1))
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
        h = h.sum(dim=[2,3])
        out = self.lin(h).squeeze(1)
        out += torch.sum(self.embed(y) * h, dim=1)
        return out


def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
def g_hinge_loss(fake_logits):
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

    z_dim = 128
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    epochs = 20
    g_losses, d_losses = [], []

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}", ncols=90)
        g_epoch_loss, d_epoch_loss = 0, 0

        for real, y in pbar:
            real = real.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Discriminator
            d_opt.zero_grad(set_to_none=True)
            z = torch.randn(real.size(0), z_dim, device=device)
            with torch.no_grad():
                fake = G(z, y)
            real_logits = D(real, y)
            fake_logits = D(fake.detach(), y)
            d_loss = d_hinge_loss(real_logits, fake_logits)
            d_loss.backward()
            d_opt.step()
            d_epoch_loss += d_loss.item()

            # Generator
            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(real.size(0), z_dim, device=device)
            fake = G(z, y)
            fake_logits = D(fake, y)
            g_loss = g_hinge_loss(fake_logits)
            g_loss.backward()
            g_opt.step()
            g_epoch_loss += g_loss.item()

            pbar.set_postfix({"d": f"{d_loss.item():.3f}", "g": f"{g_loss.item():.3f}"})

        g_losses.append(g_epoch_loss / len(dl))
        d_losses.append(d_epoch_loss / len(dl))

    # Save models
    torch.save(G.state_dict(), 'models/cgan_scratch_G_128_20epochs.pth')
    torch.save(D.state_dict(), 'models/cgan_scratch_D_128_20epochs.pth')

    # Plot and save training curves
    plt.figure(figsize=(7,5))
    plt.plot(range(1, epochs+1), d_losses, label='Discriminator Loss')
    plt.plot(range(1, epochs+1), g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/training_loss_curve.png')
    plt.close()
    print("Saved cGAN checkpoints and loss plot.")

if __name__ == "__main__":
    main()
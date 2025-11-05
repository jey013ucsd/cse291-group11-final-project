import glob
import pickle
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from tqdm import tqdm


class PKLDataset(Dataset):
    def __init__(self, pkl_paths):
        self.paths = pkl_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            obj = pickle.load(f)
        img = torch.from_numpy(obj["image"]).float() / 127.5 - 1.0
        img = img.unsqueeze(0).repeat(3, 1, 1)
        return img

# Discriminator
class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 64
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),      nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch*2, 4, 2, 1),   nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch*4, ch*8, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(ch*8*8*8, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.view(x.size(0), -1)
        return self.head(h)

# Loss
def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    target_dir = Path(cfg["target_dir"])
    img_size = int(cfg.get("IMG_SIZE", 128))

    train_pkls = sorted(glob.glob(str(target_dir / "train" / "NORMAL" / "*.pkl")))
    if not train_pkls:
        raise RuntimeError("No PKLs found at dataset/train/NORMAL/*.pkl")

    ds = PKLDataset(train_pkls)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    G = BigGAN.from_pretrained('biggan-deep-128').to(device).train()
    D = Disc().to(device).train()

    CLASS_ID = 0
    def class_vec(batch):
        v = torch.zeros(batch, 1000, device=device)
        v[:, CLASS_ID] = 1.0
        return v

    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    epochs = 5
    trunc = 0.5
    print(f"Training for {epochs} epochs with truncation {trunc}")
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}", ncols=90)
        for real in pbar:
            real = real.to(device, non_blocking=True)

            # Discriminator 
            d_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = torch.from_numpy(truncated_noise_sample(truncation=trunc, batch_size=real.size(0))).to(device).float()
                y = class_vec(real.size(0))
                fake = G(z, y, truncation=trunc)

            real_logits = D(real)
            fake_logits = D(fake.detach())
            d_loss = d_hinge_loss(real_logits, fake_logits)
            d_loss.backward()
            d_opt.step()

            # Generator 
            g_opt.zero_grad(set_to_none=True)
            z = torch.from_numpy(truncated_noise_sample(truncation=trunc, batch_size=real.size(0))).to(device).float()
            y = class_vec(real.size(0))
            fake = G(z, y, truncation=trunc)
            fake_logits = D(fake)
            g_loss = g_hinge_loss(fake_logits)
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix({"d_loss": f"{d_loss.item():.3f}", "g_loss": f"{g_loss.item():.3f}"})

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out = models_dir / "biggan_finetuned_normal_128.pth"
    torch.save(G.state_dict(), out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()

"""
Evaluate Unconditional Baselines (Unconditional GAN and VAE)
Uses KID metric for more stable evaluation with small sample sizes
"""

import os
import glob
import pickle
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from torchvision import models
warnings.filterwarnings('ignore')


# ============================================================================
# Model Architectures
# ============================================================================

class UnconditionalGenerator(nn.Module):
    """Unconditional GAN Generator"""
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


class VAEEncoder(nn.Module):
    """VAE Encoder"""
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
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder"""
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
    """Complete VAE model"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

class InceptionV3(nn.Module):
    
    def __init__(self, resize_input=True, normalize_input=True):
        super().__init__()
        try:
            inception = models.inception_v3(pretrained=True)
        except:
            inception = models.inception_v3(weights='IMAGENET1K_V1')
        
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        if self.normalize_input:
            x = 2 * x - 1
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = self.blocks(x)
        return x.squeeze(3).squeeze(2)

def polynomial_kernel(X, Y, degree=3):
    """Polynomial kernel for KID"""
    d = X.shape[1]
    K = np.dot(X, Y.T) / d
    return (1.0 + K) ** degree


def calculate_kid(real_features, fake_features, subsample_size=None):
    """
    Calculate Kernel Inception Distance (KID)
    """
    if subsample_size is not None:
        if len(real_features) > subsample_size:
            idx = np.random.choice(len(real_features), subsample_size, replace=False)
            real_features = real_features[idx]
        if len(fake_features) > subsample_size:
            idx = np.random.choice(len(fake_features), subsample_size, replace=False)
            fake_features = fake_features[idx]
    
    real_features = real_features / (np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-10)
    fake_features = fake_features / (np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-10)
    
    K_real_real = polynomial_kernel(real_features, real_features)
    n_real = len(real_features)
    term1 = (np.sum(K_real_real) - np.trace(K_real_real)) / (n_real * (n_real - 1))
    
    K_fake_fake = polynomial_kernel(fake_features, fake_features)
    n_fake = len(fake_features)
    term3 = (np.sum(K_fake_fake) - np.trace(K_fake_fake)) / (n_fake * (n_fake - 1))
    
    K_real_fake = polynomial_kernel(real_features, fake_features)
    term2 = np.mean(K_real_fake)
    
    kid = term1 + term3 - 2 * term2
    return max(0, kid) * 1000


def extract_features(images, model, device, batch_size=50):
    """Extract InceptionV3 features from images"""
    model.eval()
    
    # Normalize images
    images = (images + 1.0) / 2.0
    
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def calculate_kid_from_images(real_images, fake_images, model, device, batch_size=50, subsample_size=None):
    """Calculate KID from image tensors"""
    real_features = extract_features(real_images, model, device, batch_size)
    fake_features = extract_features(fake_images, model, device, batch_size)
    return calculate_kid(real_features, fake_features, subsample_size=subsample_size)


# ============================================================================
# Dataset Loading
# ============================================================================

class PKLDataset(Dataset):
    """Load all real images from PKL files (both classes combined)"""
    def __init__(self, root, split='test', max_samples=None):
        self.samples = []
        for label_name in ["NORMAL", "PNEUMONIA"]:
            pattern = str(Path(root) / split / label_name / "*.pkl")
            self.samples.extend(glob.glob(pattern))
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        if not self.samples:
            raise RuntimeError(f"No PKL files found in {root}/{split}/")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with open(self.samples[idx], "rb") as f:
            obj = pickle.load(f)
        x = torch.from_numpy(obj["image"]).float() / 127.5 - 1.0
        return x.unsqueeze(0)


class PKLDatasetByClass(Dataset):
    """Load real images from PKL files for a specific class"""
    def __init__(self, root, split='test', label='NORMAL', max_samples=None):
        self.samples = []
        pattern = str(Path(root) / split / label / "*.pkl")
        self.samples = glob.glob(pattern)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        if not self.samples:
            raise RuntimeError(f"No PKL files found in {root}/{split}/{label}/")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with open(self.samples[idx], "rb") as f:
            obj = pickle.load(f)
        x = torch.from_numpy(obj["image"]).float() / 127.5 - 1.0
        return x.unsqueeze(0)


def generate_images_unconditional_gan(model_path, num_images, device, z_dim=128, batch_size=50):
    """Generate images from unconditional GAN"""
    G = UnconditionalGenerator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    
    generated = []
    for i in range(0, num_images, batch_size):
        batch_size_curr = min(batch_size, num_images - i)
        z = torch.randn(batch_size_curr, z_dim, device=device)
        
        with torch.no_grad():
            fake = G(z)
        generated.append(fake.cpu())
    
    return torch.cat(generated, dim=0)


def generate_images_vae(model_path, num_images, device, latent_dim=128, batch_size=50):
    """Generate images from VAE"""
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    
    generated = []
    for i in range(0, num_images, batch_size):
        batch_size_curr = min(batch_size, num_images - i)
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        
        with torch.no_grad():
            fake = vae.decoder(z)
        generated.append(fake.cpu())
    
    return torch.cat(generated, dim=0)


# ============================================================================
# Visualization
# ============================================================================

def save_sample_grid(model_path, model_type, device, output_path, num_samples=9, z_dim=128, latent_dim=128):
    """
    Generate and save a 3x3 grid of sample images
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'unconditional_gan' or 'vae'
        device: Device to use
        output_path: Path to save the grid image
        num_samples: Number of samples (default 9 for 3x3 grid)
        z_dim: Latent dimension for GAN
        latent_dim: Latent dimension for VAE
    """
    print(f"\nGenerating {num_samples} sample images for visualization...")
    
    if model_type == 'unconditional_gan':
        images = generate_images_unconditional_gan(model_path, num_samples, device, z_dim=z_dim, batch_size=num_samples)
    elif model_type == 'vae':
        images = generate_images_vae(model_path, num_samples, device, latent_dim=latent_dim, batch_size=num_samples)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    images = images.cpu().numpy()
    images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    images = np.clip(images, 0, 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(f'{model_type.replace("_", " ").title()} - Sample Generations', fontsize=16, y=0.98)
    
    for idx in range(num_samples):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        ax.imshow(images[idx, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample grid saved to {output_path}")


def evaluate_baseline(model_path, model_type, data_dir, num_samples=500, batch_size=50, device='cuda', output_dir=None):
    """
    Evaluate unconditional baseline model
    Calculates KID separately for NORMAL and PNEUMONIA classes
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'unconditional_gan' or 'vae'
        data_dir: Path to dataset
        num_samples: Number of samples to generate (total, will compare against num_samples per class)
        batch_size: Batch size for processing
        device: Device to use
        output_dir: Directory to save results
    """
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_type.upper()}: {model_path}")
    print(f"{'='*80}\n")
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\nLoading InceptionV3 for feature extraction...")
    inception = InceptionV3().to(device)
    inception.eval()
    
    print(f"\nGenerating {num_samples} fake images...")
    if model_type == 'unconditional_gan':
        fake_images = generate_images_unconditional_gan(model_path, num_samples, device, batch_size=batch_size)
    elif model_type == 'vae':
        fake_images = generate_images_vae(model_path, num_samples, device, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"Generated {len(fake_images)} fake images")
    
    results = {
        'model_path': str(model_path),
        'model_type': model_type,
        'num_samples': num_samples,
    }

    class_names = ['NORMAL', 'PNEUMONIA']
    
    for class_name in class_names:
        print(f"\n{'='*60}")
        print(f"Evaluating against {class_name} class")
        print(f"{'='*60}")
        
        print(f"\nLoading real {class_name} images...")
        real_dataset = PKLDatasetByClass(data_dir, split='test', label=class_name, max_samples=num_samples)
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        real_images = torch.cat([batch for batch in real_loader], dim=0)[:num_samples]
        print(f"Loaded {len(real_images)} real {class_name} images")
        
        print(f"\nCalculating KID vs {class_name}...")
        kid = calculate_kid_from_images(real_images, fake_images, inception, device, batch_size, subsample_size=None)
        print(f"KID ({class_name}): {kid:.4f}")
        
        results[f'kid_{class_name.lower()}'] = float(kid)
    
    results['kid_combined'] = (results['kid_normal'] + results['kid_pneumonia']) / 2
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        grid_path = output_dir / f"samples_{model_type}_{Path(model_path).stem}.png"
        save_sample_grid(model_path, model_type, device, grid_path)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Model Type: {model_type}")
    print(f"\n{'Class':<15} | {'KID Score':<12}")
    print("-" * 30)
    print(f"{'NORMAL':<15} | {results['kid_normal']:<12.4f}")
    print(f"{'PNEUMONIA':<15} | {results['kid_pneumonia']:<12.4f}")
    print("-" * 30)
    print(f"{'AVERAGE':<15} | {results['kid_combined']:<12.4f}")
    print(f"{'='*80}\n")
    
    if output_dir:
        result_file = output_dir / f"metrics_{model_type}_{Path(model_path).stem}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Unconditional Baselines (GAN & VAE)")
    
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['unconditional_gan', 'vae'],
                        help='Model type: unconditional_gan or vae')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/mps/cpu)')
    parser.add_argument('--output_dir', type=str, default='results_new/metrics_baselines', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_baseline(
        model_path=args.model,
        model_type=args.model_type,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


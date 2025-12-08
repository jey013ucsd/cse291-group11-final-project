"""
Evaluation Metrics by Class for Medical Imaging GANs
Calculates separate FID and Inception Score for NORMAL and PNEUMONIA classes
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
from scipy import linalg
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torchvision import models

class Generator(nn.Module):
    """Conditional GAN Generator"""
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


# ============================================================================

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


# ============================================================================
# FID Calculation
# ============================================================================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two multivariate Gaussians"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def compute_statistics(images, model, batch_size, device):
    """Compute mean and covariance of inception features"""
    model.eval()
    
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_fid(real_images, fake_images, model, device, batch_size=50):
    """Calculate FID between real and fake images"""
    real_images = (real_images + 1.0) / 2.0
    fake_images = (fake_images + 1.0) / 2.0
    
    m1, s1 = compute_statistics(real_images, model, batch_size, device)
    m2, s2 = compute_statistics(fake_images, model, batch_size, device)
    
    return calculate_frechet_distance(m1, s1, m2, s2)


# ============================================================================
# KID (Kernel Inception Distance) Calculation
# More stable than FID for small sample sizes
# ============================================================================

def polynomial_kernel(X, Y, degree=3):
    """
    Polynomial kernel as used in KID: k(x,y) = (1 + <x,y>/d)^degree
    where d is the feature dimension
    
    Args:
        X: [N, D] feature matrix
        Y: [M, D] feature matrix
        degree: Polynomial degree (default 3)
    
    Returns:
        [N, M] kernel matrix
    """
    d = X.shape[1]  # Feature dimension
    K = np.dot(X, Y.T) / d
    return (1.0 + K) ** degree


def calculate_kid(real_features, fake_features, subsample_size=None):
    """
    Calculate Kernel Inception Distance (KID)
    
    KID = MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    
    Args:
        real_features: [N, D] numpy array of real image features
        fake_features: [M, D] numpy array of fake image features
        subsample_size
    
    Returns:
        KID score (float)
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


def calculate_kid_from_images(real_images, fake_images, model, device, batch_size=50, subsample_size=None):
    """
    Calculate KID from image tensors
    
    Args:
        real_images: Tensor of real images [N, C, H, W]
        fake_images: Tensor of fake images [M, C, H, W]
        model: InceptionV3 feature extractor
        device: Device to run on
        batch_size: Batch size for processing
        subsample_size: Optional subsampling for KID calculation
    
    Returns:
        KID score (float)
    """
    model.eval()
    
    real_images = (real_images + 1.0) / 2.0
    fake_images = (fake_images + 1.0) / 2.0
    
    real_features = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model(batch)
        real_features.append(feat.cpu().numpy())
    real_features = np.concatenate(real_features, axis=0)
    
    fake_features = []
    for i in range(0, len(fake_images), batch_size):
        batch = fake_images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model(batch)
        fake_features.append(feat.cpu().numpy())
    fake_features = np.concatenate(fake_features, axis=0)
    
    return calculate_kid(real_features, fake_features, subsample_size=subsample_size)


# Inception Score Calculation

def calculate_inception_score(images, device, batch_size=50, splits=10):
    try:
        inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    except:
        inception_model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
    
    inception_model.eval()
    
    images = (images + 1.0) / 2.0
    
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            pred = F.softmax(inception_model(batch), dim=1).cpu().numpy()
        preds.append(pred)
    
    preds = np.concatenate(preds, axis=0)
    
    split_scores = []
    n = len(preds)
    for k in range(splits):
        part = preds[k * (n // splits): (k + 1) * (n // splits), :]
        py = np.mean(part, axis=0)
        scores = [np.sum(pyx * np.log(pyx / py + 1e-10)) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)



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


def generate_images_by_class(model_path, num_images, class_label, device, z_dim=128, batch_size=50):
    G = Generator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    
    generated = []
    
    for i in range(0, num_images, batch_size):
        batch_size_curr = min(batch_size, num_images - i)
        z = torch.randn(batch_size_curr, z_dim, device=device)
        y = torch.full((batch_size_curr,), class_label, dtype=torch.long, device=device)
        
        with torch.no_grad():
            fake = G(z, y)
        generated.append(fake.cpu())
    
    return torch.cat(generated, dim=0)



def evaluate_by_class(model_path, data_dir, num_samples=500, batch_size=50, device='cuda', output_dir=None):
    print(f"Evaluating by Class: {model_path}")

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load InceptionV3
    inception = InceptionV3().to(device)
    inception.eval()
    
    results = {
        'model_path': str(model_path),
        'num_samples_per_class': num_samples,
    }
    
    class_names = ['NORMAL', 'PNEUMONIA']
    class_labels = [0, 1]
    
    for class_name, class_label in zip(class_names, class_labels):
        print(f"\n{'='*60}")
        print(f"Evaluating {class_name} class (label={class_label})")
        print(f"{'='*60}")
        
        print(f"\nLoading real {class_name} images...")
        real_dataset = PKLDatasetByClass(data_dir, split='test', label=class_name, max_samples=num_samples)
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        real_images = torch.cat([batch for batch in real_loader], dim=0)[:num_samples]
        print(f"Loaded {len(real_images)} real {class_name} images")
        
        print(f"\nGenerating {num_samples} fake {class_name} images...")
        fake_images = generate_images_by_class(model_path, num_samples, class_label, device, batch_size=batch_size)
        print(f"Generated {len(fake_images)} fake {class_name} images")
        
        print(f"\nCalculating FID for {class_name}...")
        fid = calculate_fid(real_images, fake_images, inception, device, batch_size)
        print(f"FID ({class_name}): {fid:.4f}")
        
        print(f"\nCalculating KID for {class_name}...")
        kid = calculate_kid_from_images(real_images, fake_images, inception, device, batch_size, subsample_size=None)
        print(f"KID ({class_name}): {kid:.4f}")
        
        print(f"\nCalculating Inception Score for {class_name}...")
        is_mean, is_std = calculate_inception_score(fake_images, device, batch_size)
        print(f"IS ({class_name}): {is_mean:.4f} ± {is_std:.4f}")
        
        results[f'fid_{class_name.lower()}'] = float(fid)
        results[f'kid_{class_name.lower()}'] = float(kid)
        results[f'is_mean_{class_name.lower()}'] = float(is_mean)
        results[f'is_std_{class_name.lower()}'] = float(is_std)
    

    
    results['fid_combined'] = (results['fid_normal'] + results['fid_pneumonia']) / 2
    results['kid_combined'] = (results['kid_normal'] + results['kid_pneumonia']) / 2
    results['is_mean_combined'] = (results['is_mean_normal'] + results['is_mean_pneumonia']) / 2
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Class':<15} | {'FID Score':<12} | {'KID Score':<12} | {'Inception Score':<20}")
    print("-" * 70)
    print(f"{'NORMAL':<15} | {results['fid_normal']:<12.4f} | {results['kid_normal']:<12.4f} | {results['is_mean_normal']:.4f} ± {results['is_std_normal']:.4f}")
    print(f"{'PNEUMONIA':<15} | {results['fid_pneumonia']:<12.4f} | {results['kid_pneumonia']:<12.4f} | {results['is_mean_pneumonia']:.4f} ± {results['is_std_pneumonia']:.4f}")
    print("-" * 70)
    print(f"{'AVERAGE':<15} | {results['fid_combined']:<12.4f} | {results['kid_combined']:<12.4f} | {results['is_mean_combined']:.4f}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        result_file = output_dir / f"metrics_by_class_{Path(model_path).stem}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {result_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAN by class (NORMAL vs PNEUMONIA)")
    
    parser.add_argument('--model', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/mps/cpu)')
    parser.add_argument('--output_dir', type=str, default='results_new/metrics', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_by_class(
        model_path=args.model,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


"""
Evaluation Metrics for Medical Imaging GANs
Calculates FID and Inception Score for generated images
"""

import os
import glob
import pickle
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy import linalg
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Generator Architecture (same as in federated_learning.py)
# ============================================================================

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
# InceptionV3 for Feature Extraction
# ============================================================================

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 for feature extraction (FID & IS)"""
    
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    
    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True):
        super().__init__()
        
        try:
            from torchvision import models
            inception = models.inception_v3(pretrained=True)
        except:
            print("Warning: Could not load pretrained InceptionV3. Install torchvision>=0.6.0")
            from torchvision import models
            inception = models.inception_v3(weights='IMAGENET1K_V1')
        
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        
        self.blocks = nn.ModuleList()
        
        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        
        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))
        
        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        
        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        output = []
        
        # Resize to 299x299 for InceptionV3
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize input
        if self.normalize_input:
            x = 2 * x - 1  # Scale from [0, 1] to [-1, 1]
        
        # Convert grayscale to RGB by repeating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                output.append(x)
            
            if idx == self.last_needed_block:
                break
        
        return output


# ============================================================================
# FID Score Calculation
# ============================================================================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two multivariate Gaussians
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f'FID calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_statistics_of_images(images, model, batch_size=50, device='cuda'):
    """Compute mean and covariance of inception features"""
    model.eval()
    
    n_images = len(images)
    n_batches = (n_images + batch_size - 1) // batch_size
    
    pred_arr = []
    
    for i in tqdm(range(n_batches), desc="Computing features"):
        start = i * batch_size
        end = min(start + batch_size, n_images)
        batch = images[start:end].to(device)
        
        with torch.no_grad():
            pred = model(batch)[0]
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
    
    pred_arr = np.concatenate(pred_arr, axis=0)
    
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    return mu, sigma


def calculate_fid(real_images, fake_images, device='cuda', batch_size=50):
    """
    Calculate FID score between real and fake images
    
    Args:
        real_images: Tensor of real images [N, C, H, W]
        fake_images: Tensor of fake images [N, C, H, W]
        device: Device to run computation
        batch_size: Batch size for processing
    
    Returns:
        FID score (float)
    """
    # Load InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    
    # Normalize images to [0, 1]
    real_images = (real_images + 1.0) / 2.0
    fake_images = (fake_images + 1.0) / 2.0
    
    # Compute statistics
    print("Computing statistics for real images...")
    m1, s1 = compute_statistics_of_images(real_images, model, batch_size, device)
    
    print("Computing statistics for fake images...")
    m2, s2 = compute_statistics_of_images(fake_images, model, batch_size, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


# ============================================================================
# Inception Score Calculation
# ============================================================================

def calculate_inception_score(images, device='cuda', batch_size=50, splits=10):
    """
    Calculate Inception Score
    
    IS = exp(E[KL(p(y|x) || p(y))])
    
    Args:
        images: Generated images [N, C, H, W]
        device: Device to run computation
        batch_size: Batch size for processing
        splits: Number of splits for computing mean and std
    
    Returns:
        (mean, std) of Inception Score
    """
    from torchvision import models
    
    # Load InceptionV3 with classifier head
    try:
        inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    except:
        inception_model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
    
    inception_model.eval()
    
    # Normalize images to [0, 1]
    images = (images + 1.0) / 2.0
    
    n_images = len(images)
    
    # Get predictions
    preds = []
    
    for i in tqdm(range(0, n_images, batch_size), desc="Computing Inception Score"):
        batch = images[i:i+batch_size].to(device)
        
        # Convert grayscale to RGB
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        
        # Resize to 299x299
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            pred = F.softmax(inception_model(batch), dim=1).cpu().numpy()
        
        preds.append(pred)
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute score
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (n_images // splits): (k + 1) * (n_images // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


# ============================================================================
# Dataset Loading
# ============================================================================

class PKLDataset(Dataset):
    """Load real images from PKL files"""
    def __init__(self, root, split='test', max_samples=None):
        self.samples = []
        for label_name in ["NORMAL", "PNEUMONIA"]:
            pattern = str(Path(root) / split / label_name / "*.pkl")
            for p in glob.glob(pattern):
                self.samples.append(p)
        
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


def generate_images(model_path, num_images, device, z_dim=128, batch_size=50):
    """Generate images using trained generator"""
    G = Generator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    
    generated = []
    
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating images"):
        batch_size_curr = min(batch_size, num_images - i * batch_size)
        
        z = torch.randn(batch_size_curr, z_dim, device=device)
        # Generate equal mix of both classes
        y = torch.tensor([i % 2 for i in range(batch_size_curr)], device=device)
        
        with torch.no_grad():
            fake = G(z, y)
        
        generated.append(fake.cpu())
    
    return torch.cat(generated, dim=0)


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_model(
    model_path,
    data_dir,
    num_samples=1000,
    batch_size=50,
    device='cuda',
    output_dir=None
):
    """
    Evaluate a trained GAN model
    
    Args:
        model_path: Path to generator checkpoint
        data_dir: Path to dataset directory
        num_samples: Number of images to generate for evaluation
        batch_size: Batch size for computation
        device: Device to run on
        output_dir: Optional directory to save results
    
    Returns:
        Dictionary with FID and IS scores
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*80}\n")
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load real images
    print(f"\nLoading real images from {data_dir}...")
    real_dataset = PKLDataset(data_dir, split='test', max_samples=num_samples)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    real_images = []
    for batch in tqdm(real_loader, desc="Loading real images"):
        real_images.append(batch)
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    print(f"Loaded {len(real_images)} real images")
    
    # Generate fake images
    print(f"\nGenerating {num_samples} fake images...")
    fake_images = generate_images(model_path, num_samples, device, batch_size=batch_size)
    print(f"Generated {len(fake_images)} fake images")
    
    # Calculate FID
    print("\n" + "="*80)
    print("Calculating FID Score...")
    print("="*80)
    fid_score = calculate_fid(real_images, fake_images, device=device, batch_size=batch_size)
    print(f"\n✓ FID Score: {fid_score:.4f}")
    
    # Calculate Inception Score
    print("\n" + "="*80)
    print("Calculating Inception Score...")
    print("="*80)
    is_mean, is_std = calculate_inception_score(fake_images, device=device, batch_size=batch_size)
    print(f"\n✓ Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    results = {
        'model_path': str(model_path),
        'num_samples': num_samples,
        'fid_score': float(fid_score),
        'inception_score_mean': float(is_mean),
        'inception_score_std': float(is_std)
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        import json
        result_file = output_dir / f"metrics_{Path(model_path).stem}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {result_file}")
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GAN models with FID and Inception Score"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to generator checkpoint (.pth file)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset',
        help='Path to dataset directory (default: dataset)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to generate for evaluation (default: 1000)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for computation (default: 50)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/metrics',
        help='Directory to save results (default: results/metrics)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path=args.model,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


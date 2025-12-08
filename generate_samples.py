"""
Generate sample images using trained GAN models
Creates example images for both NORMAL and PNEUMONIA classes
"""

import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
import matplotlib.pyplot as plt
import numpy as np
import argparse


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


def load_generator(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a trained generator model"""
    G = Generator(z_dim=128, num_classes=2, embed_dim=50, base_ch=512, out_ch=1)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.to(device)
    G.eval()
    return G


def generate_samples(G, num_samples=5, z_dim=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Generate sample images for both classes"""
    with torch.no_grad():
        # normal (class 0) images
        z_normal = torch.randn(num_samples, z_dim, device=device)
        y_normal = torch.zeros(num_samples, dtype=torch.long, device=device)
        imgs_normal = G(z_normal, y_normal)
        
        # pneumonia (class 1) images
        z_pneumonia = torch.randn(num_samples, z_dim, device=device)
        y_pneumonia = torch.ones(num_samples, dtype=torch.long, device=device)
        imgs_pneumonia = G(z_pneumonia, y_pneumonia)
        
    return imgs_normal.cpu(), imgs_pneumonia.cpu()


def save_image_grid(imgs_normal, imgs_pneumonia, output_path, num_samples=5):
    """Create and save two separate grids of generated images"""
    base_path = output_path.rsplit('.', 1)
    if len(base_path) == 2:
        normal_path = f"{base_path[0]}_normal.{base_path[1]}"
        pneumonia_path = f"{base_path[0]}_pneumonia.{base_path[1]}"
    else:
        normal_path = f"{output_path}_normal.png"
        pneumonia_path = f"{output_path}_pneumonia.png"
    
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    # normal images
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = imgs_normal[i].squeeze().numpy()
        img = (img + 1) / 2
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Generated NORMAL Chest X-rays', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(normal_path, dpi=150, bbox_inches='tight')
    print(f"Saved NORMAL images to: {normal_path}")
    plt.close()
    
    # pneumonia images
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = imgs_pneumonia[i].squeeze().numpy()
        img = (img + 1) / 2
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Generated PNEUMONIA Chest X-rays', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(pneumonia_path, dpi=150, bbox_inches='tight')
    print(f"Saved PNEUMONIA images to: {pneumonia_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate sample images from trained GAN')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the generator model (.pth file)')
    parser.add_argument('--output', type=str, default='generated_samples.png',
                        help='Output path for the generated image grid')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from: {args.model}")
    G = load_generator(args.model, device)
    print("Model loaded successfully")
    print(f"Generating {args.num_samples} samples per class...")
    imgs_normal, imgs_pneumonia = generate_samples(G, args.num_samples, device=device)
    print("Samples generated")
    
    save_image_grid(imgs_normal, imgs_pneumonia, args.output, args.num_samples)
    print(f"\nDone! Generated {args.num_samples} NORMAL and {args.num_samples} PNEUMONIA images.")


if __name__ == "__main__":
    main()


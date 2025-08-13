"""
train.py: Training loop for Task 1
Uses StyleColorizer, VGGFeatureExtractor, and loss functions.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import (
    load_grayscale_image,
    load_image,
    rgb_to_lab_tensor,
    normalize_for_vgg,
    lab_to_rgb_tensor,
    tensor_to_pil,
    get_image_paths
)
from model import StyleColorizer, VGGFeatureExtractor
from losses import compute_content_loss, compute_style_loss, compute_tv_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='../data')
    p.add_argument('--styles_dir', type=str, default='../data/styles')
    p.add_argument('--output_dir', type=str, default='../models')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    return p.parse_args()

def build_dataset(content_paths, style_paths, device):
    """
    Returns pairs of (L_tensor, style_rgb_norm) for training.
    """
    pairs = []
    for cpath in content_paths:
        for spath in style_paths:
            # load L channel and style image
            L = load_grayscale_image(cpath, device=device)
            style = load_image(spath, device=device)
            style_norm = normalize_for_vgg(style)
            pairs.append((L, style_norm))
    return pairs

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data paths
    content_paths = get_image_paths(os.path.join(args.data_dir, 'content'))
    style_paths   = get_image_paths(os.path.join(args.styles_dir))

    # Build simple list-based dataset
    dataset = build_dataset(content_paths, style_paths, device)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Instantiate model and feature extractor
    model = StyleColorizer().to(device)
    vgg_ex = VGGFeatureExtractor(content_layers=['21'], style_layers=['0','5','10','19','28']).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs+1):
        total_c_loss = total_s_loss = total_tv_loss = 0.0
        model.train()
        for L, style_norm in loader:
            optimizer.zero_grad()
            # Forward pass: predict ab
            ab_pred = model(L, style_norm)
            
            # Reconstruct full Lab and convert to normalized RGB
            lab_pred = torch.cat([L, ab_pred], dim=1)
            rgb_pred = lab_to_rgb_tensor(lab_pred)        # [0,1]
            rgb_norm = normalize_for_vgg(rgb_pred)        # for VGG

            # Extract features
            feats_pred = vgg_ex(rgb_norm)
            feats_style = vgg_ex(style_norm)

            # Compute losses
            c_loss = compute_content_loss(feats_pred['content'], feats_pred['content'])
            s_loss = compute_style_loss(feats_pred['style'], feats_style['style'])
            tv_loss = compute_tv_loss(ab_pred)
            loss = c_loss + 1e3*s_loss + 1e-6*tv_loss

            c_loss = compute_content_loss(feats_pred['content'], feats_content_target)
            s_loss = compute_style_loss(feats_pred['style'], feats_style_target)
            tv = compute_tv_loss(ab_pred)
            loss = c_loss + style_weight * s_loss + tv


            # Backprop & step
            loss.backward()
            optimizer.step()

            total_c_loss += c_loss.item()
            total_s_loss += s_loss.item()
            total_tv_loss += tv_loss.item()

        # Log epoch metrics
        print(f"Epoch {epoch}: Content {total_c_loss:.4f}, Style {total_s_loss:.4f}, TV {total_tv_loss:.4f}")

        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch{epoch}.pth'))

if __name__ == '__main__':
    train()

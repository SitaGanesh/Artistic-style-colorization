"""
inference.py: Single-image inference pipeline.
Loads a trained checkpoint and colorizes one image.
"""

import os
import torch
from data_utils import (
    load_grayscale_image,
    load_image,
    normalize_for_vgg,
    lab_to_rgb_tensor,
    tensor_to_pil
)
from model import StyleColorizer

def colorize_image(
    checkpoint_path,
    content_path,
    style_path,
    output_path,
    device='cpu',
    use_direct_color_transfer=True  # New parameter
):
    # Setup
    device = torch.device(device)
    model = StyleColorizer().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load inputs
    L = load_grayscale_image(content_path, device=device)          # (1,1,H,W)
    style_rgb = load_image(style_path, device=device)              # (1,3,H,W)
    style_norm = normalize_for_vgg(style_rgb)

    # Predict ab channels
    with torch.no_grad():
        ab_pred = model(L, style_norm)
        
        # Handle shape mismatch
        _, _, H, W = L.shape
        if ab_pred.shape[2:] != (H, W):
            ab_pred = torch.nn.functional.interpolate(
                ab_pred, size=(H, W), mode='bilinear', align_corners=False
            )
        
        # Apply direct color transfer from style image
        if use_direct_color_transfer:
            from data_utils import get_lab_mean_std
            a_mean, a_std, b_mean, b_std = get_lab_mean_std(style_path)
            
            # Normalize ab_pred to zero mean unit std, then re-scale
            a = ab_pred[0,0]
            b = ab_pred[0,1]
            
            a = (a - a.mean()) / (a.std() + 1e-5) * a_std + a_mean
            b = (b - b.mean()) / (b.std() + 1e-5) * b_std + b_mean
            
            ab_pred = torch.stack([a, b], dim=0).unsqueeze(0)

    # Combine L + ab, convert to RGB
    lab_full = torch.cat([L, ab_pred], dim=1)
    rgb_pred = lab_to_rgb_tensor(lab_full)

    # Convert to PIL and save
    out_img = tensor_to_pil(rgb_pred)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_img.save(output_path)
    print(f"Saved colorized image: {output_path}")


if __name__ == '__main__':
    # Example usage:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--content', required=True)
    p.add_argument('--style', required=True)
    p.add_argument('--output', default='results/output.png')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    colorize_image(
        checkpoint_path=args.checkpoint,
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        device=args.device
    )

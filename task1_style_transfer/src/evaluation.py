"""
evaluation.py: Comprehensive evaluation metrics for Task 1
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.color import lab2rgb, rgb2lab

def compute_psnr(img1, img2, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def compute_ssim(img1, img2):
    """Structural Similarity Index Measure"""
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.squeeze().permute(1,2,0).cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.squeeze().permute(1,2,0).cpu().numpy()
    
    # Ensure images are in [0,1] range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    return ssim(img1, img2, data_range=1.0, channel_axis=2)  # Fixed: removed multichannel

def compute_content_fidelity(original_gray, colorized_rgb):
    """Content Fidelity: how well structure is preserved"""
    # Convert colorized back to grayscale for comparison
    colorized_gray = cv2.cvtColor(
        (colorized_rgb * 255).astype(np.uint8), 
        cv2.COLOR_RGB2GRAY
    ) / 255.0
    
    if torch.is_tensor(original_gray):
        original_gray = original_gray.squeeze().cpu().numpy()
    
    return ssim(original_gray, colorized_gray, data_range=1.0)

def compute_style_effectiveness(style_img, colorized_img, vgg_extractor):
    """Style Effectiveness: how well style is transferred"""
    with torch.no_grad():
        style_features = vgg_extractor(style_img)['style']
        colorized_features = vgg_extractor(colorized_img)['style']
        
        style_score = 0
        for s_feat, c_feat in zip(style_features, colorized_features):
            # Gram matrix similarity
            s_gram = gram_matrix(s_feat)
            c_gram = gram_matrix(c_feat)
            style_score += F.mse_loss(s_gram, c_gram)
    
    # Convert to similarity (lower loss = higher effectiveness)
    return 1.0 / (1.0 + style_score.item())

def gram_matrix(feature):
    """Compute Gram matrix for style comparison"""
    b, c, h, w = feature.size()
    f = feature.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

def evaluate_colorization_quality(
    original_gray, 
    style_img, 
    colorized_img, 
    vgg_extractor,
    ground_truth_rgb=None
):
    """
    Comprehensive evaluation of colorization quality
    Returns dict with all metrics
    """
    metrics = {}
    
    # Content fidelity 
    try:
        metrics['content_fidelity'] = compute_content_fidelity(
            original_gray, colorized_img.squeeze().permute(1,2,0).cpu().numpy()
        )
    except Exception as e:
        print(f"Content fidelity error: {e}")
        metrics['content_fidelity'] = 0.0
    
    # Style effectiveness
    try:
        metrics['style_effectiveness'] = compute_style_effectiveness(
            style_img, colorized_img, vgg_extractor
        )
    except Exception as e:
        print(f"Style effectiveness error: {e}")
        metrics['style_effectiveness'] = 0.0
    
    # If ground truth available, compute accuracy metrics
    if ground_truth_rgb is not None:
        try:
            metrics['psnr'] = compute_psnr(colorized_img, ground_truth_rgb).item()
            metrics['ssim'] = compute_ssim(colorized_img, ground_truth_rgb)
        except Exception as e:
            print(f"PSNR/SSIM error: {e}")
            metrics['psnr'] = 0.0
            metrics['ssim'] = 0.0
    
    # Color diversity (measures if output is actually colorized)
    try:
        colorized_np = colorized_img.squeeze().permute(1,2,0).cpu().numpy()
        lab_img = rgb2lab(colorized_np)
        ab_channels = lab_img[:,:,1:]  # a and b channels
        metrics['color_diversity'] = np.std(ab_channels)
    except Exception as e:
        print(f"Color diversity error: {e}")
        metrics['color_diversity'] = 0.0
    
    return metrics

class ModelEvaluator:
    """Complete model evaluation pipeline"""
    
    def __init__(self, model, vgg_extractor, device='cpu'):
        self.model = model.eval()
        self.vgg_extractor = vgg_extractor
        self.device = device
        
    def evaluate_dataset(self, test_loader, style_images):
        """Evaluate model on entire test dataset"""
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (gray_imgs, rgb_imgs) in enumerate(test_loader):
                gray_imgs = gray_imgs.to(self.device)
                rgb_imgs = rgb_imgs.to(self.device)
                
                for style_idx, style_img in enumerate(style_images):
                    style_tensor = style_img.to(self.device)
                    
                    # Generate colorization
                    colorized = self.model(gray_imgs, style_tensor)
                    
                    # Handle shape mismatch
                    if colorized.shape[2:] != gray_imgs.shape[2:]:
                        colorized = F.interpolate(
                            colorized, size=gray_imgs.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    
                    # Reconstruct full image
                    lab_full = torch.cat([gray_imgs, colorized], dim=1)
                    from data_utils import lab_to_rgb_tensor
                    rgb_pred = lab_to_rgb_tensor(lab_full)
                    
                    # Evaluate each image in batch
                    for i in range(gray_imgs.size(0)):
                        metrics = evaluate_colorization_quality(
                            gray_imgs[i], style_tensor[0], 
                            rgb_pred[i:i+1], self.vgg_extractor,
                            rgb_imgs[i:i+1]
                        )
                        metrics['batch_idx'] = batch_idx
                        metrics['image_idx'] = i
                        metrics['style_idx'] = style_idx
                        all_metrics.append(metrics)
        
        return all_metrics
    
    def print_summary(self, metrics_list):
        """Print evaluation summary"""
        if not metrics_list:
            print("No metrics to summarize")
            return
            
        # FIXED: Get keys from first item in list, not the list itself
        keys = metrics_list[0].keys()
        summary = {}
        
        for key in keys:
            if key not in ['batch_idx', 'image_idx', 'style_idx']:
                values = [m[key] for m in metrics_list if key in m and m[key] is not None]
                if values:
                    summary[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        print("\n=== MODEL EVALUATION SUMMARY ===")
        for metric, stats in summary.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

# Simplified evaluation function for notebook use
def simple_evaluate_model(model, vgg_extractor, content_path, style_path, device='cpu'):
    """Simple evaluation for single image pair"""
    from data_utils import load_grayscale_image, load_image, normalize_for_vgg, lab_to_rgb_tensor
    
    # Load images
    gray_img = load_grayscale_image(content_path, device=device)
    style_img = load_image(style_path, device=device)
    style_norm = normalize_for_vgg(style_img)
    
    # Generate colorization
    model.eval()
    with torch.no_grad():
        ab_pred = model(gray_img, style_norm)
        
        # Handle shape mismatch
        if ab_pred.shape[2:] != gray_img.shape[2:]:
            ab_pred = F.interpolate(ab_pred, size=gray_img.shape[2:], mode='bilinear', align_corners=False)
        
        # Reconstruct RGB
        lab_full = torch.cat([gray_img, ab_pred], dim=1)
        rgb_pred = lab_to_rgb_tensor(lab_full)
    
    # Evaluate
    metrics = evaluate_colorization_quality(gray_img, style_norm, rgb_pred, vgg_extractor)
    
    return rgb_pred, metrics

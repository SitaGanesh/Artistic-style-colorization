"""
losses.py: Loss functions for Task 1 —
Content, style, and total variation (TV) losses for artistic style transfer.
"""

import torch
import torch.nn.functional as F

def compute_content_loss(generated_feat, target_feat):
    """
    Content loss preserves structural information from the original image.
    
    Computes Mean Squared Error (MSE) between feature maps extracted from:
    - Generated (colorized) image 
    - Target content image
    
    This ensures the colorized output maintains the same structural content
    as the input grayscale image, preventing distortion of shapes and edges.
    
    Args:
        generated_feat (torch.Tensor): VGG feature map from model's output image
        target_feat (torch.Tensor): VGG feature map from content/style reference
    
    Returns:
        torch.Tensor: Scalar content loss value for backpropagation
    """
    # Detach target to prevent gradients flowing into the reference image
    # We only want to update our model, not modify the target features
    return F.mse_loss(generated_feat, target_feat.detach())

def gram_matrix(feature):
    """
    Compute Gram matrix to capture style information from feature maps.
    
    The Gram matrix captures correlations between different feature channels,
    representing the "style" or texture information independent of spatial layout.
    This is the core mathematical operation behind neural style transfer.
    
    Process:
    1. Flatten spatial dimensions (H×W) into a single dimension
    2. Compute correlation matrix between all channel pairs
    3. Normalize by total number of elements to make scale-invariant
    
    Args:
        feature (torch.Tensor): Input feature map of shape (N, C, H, W)
                               N=batch, C=channels, H=height, W=width
    
    Returns:
        torch.Tensor: Gram matrix of shape (N, C, C) where each (C,C) matrix
                     contains correlations between all channel pairs
    """
    N, C, H, W = feature.size()
    
    # Reshape: (N, C, H, W) → (N, C, H*W)
    # This flattens spatial dimensions while keeping batch and channel separate
    f = feature.view(N, C, H * W)
    
    # Batch matrix multiplication: (N, C, H*W) × (N, H*W, C) → (N, C, C)
    # For each batch, compute correlation between all channel pairs
    G = torch.bmm(f, f.transpose(1, 2))
    
    # Normalize by total elements per feature map to make scale-invariant
    # This prevents larger images from having disproportionately large Gram values
    return G.div(C * H * W)

def compute_style_loss(generated_feats, style_feats):
    """
    Style loss transfers artistic characteristics from reference to generated image.
    
    Compares Gram matrices (texture/style representations) between:
    - Generated image features at multiple VGG layers
    - Style reference image features at the same layers
    
    Using multiple layers captures style at different scales:
    - Early layers: fine textures, brush strokes
    - Later layers: larger patterns, color schemes
    
    Args:
        generated_feats (list[torch.Tensor]): Feature maps from generated image
                                            extracted at multiple VGG layers
        style_feats (list[torch.Tensor]): Feature maps from style reference
                                        extracted at the same VGG layers
    
    Returns:
        torch.Tensor: Scalar style loss summed across all layers
    """
    loss = 0.0
    
    # Iterate through corresponding feature maps from each VGG layer
    for gen, sty in zip(generated_feats, style_feats):
        # Compute Gram matrices for both generated and style features
        Gg = gram_matrix(gen)           # Generated image style representation
        Gs = gram_matrix(sty).detach()  # Style reference (detached from gradients)
        
        # Add MSE between Gram matrices to total style loss
        # This encourages generated image to have similar texture correlations
        loss += F.mse_loss(Gg, Gs)
    
    return loss

def compute_tv_loss(ab_pred, weight=1e-6):
    """
    Total Variation (TV) loss encourages spatial smoothness in predicted colors.
    
    Penalizes large differences between neighboring pixels in the predicted
    a* and b* color channels. This prevents noisy, scattered colorization
    and encourages smooth color transitions within objects.
    
    The TV loss is computed as the sum of squared differences between:
    - Horizontally adjacent pixels
    - Vertically adjacent pixels
    
    Args:
        ab_pred (torch.Tensor): Predicted chrominance channels of shape (N, 2, H, W)
                               where 2 represents a* and b* channels
        weight (float): Scaling factor to balance TV loss with other losses
    
    Returns:
        torch.Tensor: Scalar TV loss normalized by tensor size
    """
    # Compute horizontal differences: compare each pixel with its right neighbor
    # Shape: (N, 2, H, W-1) - excludes the last column since it has no right neighbor
    diff_h = ab_pred[:, :, :, 1:] - ab_pred[:, :, :, :-1]
    
    # Compute vertical differences: compare each pixel with its bottom neighbor  
    # Shape: (N, 2, H-1, W) - excludes the last row since it has no bottom neighbor
    diff_w = ab_pred[:, :, 1:, :] - ab_pred[:, :, :-1, :]
    
    # Sum of squared differences for horizontal variations
    tv_h = torch.sum(diff_h ** 2)
    
    # Sum of squared differences for vertical variations  
    tv_w = torch.sum(diff_w ** 2)
    
    # Normalize by total number of elements to make loss scale-invariant
    # This ensures consistent loss magnitude regardless of image size
    denom = ab_pred.numel()
    
    # Return weighted and normalized total variation loss
    return weight * (tv_h + tv_w) / denom

"""
model.py: Network architectures for Task 1 —
Artistic Style Transfer + Colorization

This file contains the core neural network components that work together
to transform grayscale images into colorized versions with artistic style.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature representations from pretrained VGG-19.
    
    VGG-19 is used because:
    1. It's pretrained on ImageNet, so it understands natural image features
    2. Different layers capture different levels of abstraction:
       - Early layers (0,5,10): textures, edges, simple patterns
       - Middle layers (19): more complex shapes and patterns  
       - Later layers (21): high-level semantic content
    
    This multi-scale feature extraction is crucial for style transfer because:
    - Content should be preserved at high semantic levels
    - Style (textures/patterns) should be captured at multiple scales
    """
    
    def __init__(self, content_layers, style_layers):
        super().__init__()
        
        # Load pretrained VGG-19 feature layers (excludes classifier)
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze all VGG parameters - we only use it as a fixed feature extractor
        # We don't want to modify the pretrained features during our training
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        self.vgg_layers = vgg
        self.content_layers = content_layers  # e.g., ['21'] for high-level content
        self.style_layers = style_layers      # e.g., ['0','5','10','19','28'] for multi-scale style
        
    def forward(self, x):
        """
        Forward pass through VGG layers, collecting features at specified depths.
        
        Args:
            x (torch.Tensor): Input RGB tensor normalized for VGG: (N,3,H,W)
                             Must be ImageNet-normalized: mean=[0.485,0.456,0.406], 
                             std=[0.229,0.224,0.225]
        
        Returns:
            dict: {
               'content': feature map at content layer (typically deeper layer),
               'style': [list of feature maps at each style layer]
            }
        """
        content_feats = None
        style_feats = []
        
        # Pass through each VGG layer sequentially
        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)
            
            # Collect content features (usually just one high-level layer)
            if str(idx) in self.content_layers:
                content_feats = x
                
            # Collect style features (multiple layers for multi-scale representation)  
            if str(idx) in self.style_layers:
                style_feats.append(x)
                
        return {'content': content_feats, 'style': style_feats}

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization - the key mechanism for style transfer.
    
    AdaIN works by:
    1. Normalizing content features to have zero mean and unit variance
    2. Re-scaling them to match the mean and variance of style features
    
    This effectively transfers the "style statistics" from the style image
    to the content image while preserving the spatial structure.
    
    Mathematical operation:
    AdaIN(content, style) = σ_style * (content - μ_content)/σ_content + μ_style
    
    Where μ and σ are computed per-channel across spatial dimensions.
    """
    
    def forward(self, content_feat, style_feat):
        """
        Apply adaptive instance normalization.
        
        Args:
            content_feat (torch.Tensor): Content features (N, C, H, W)
            style_feat (torch.Tensor): Style features (N, C, H, W)
            
        Returns:
            torch.Tensor: Normalized content with style statistics applied
        """
        # Compute statistics across spatial dimensions (H, W) for each channel
        # keepdim=True maintains (N, C, 1, 1) shape for broadcasting
        c_mean = content_feat.mean([2,3], keepdim=True)  # Content mean per channel
        c_std  = content_feat.std([2,3], keepdim=True)   # Content std per channel
        s_mean = style_feat.mean([2,3], keepdim=True)    # Style mean per channel  
        s_std  = style_feat.std([2,3], keepdim=True)     # Style std per channel
        
        # Debug prints to monitor style statistics during training
        print(f"AdaIN: style a_std {s_std[:,0,0,0].item():.3f}, b_std {s_std[:,1,0,0].item():.3f}")
        print(f"AdaIN: style a_mean {s_mean[:,0,0,0].item():.3f}, b_mean {s_mean[:,1,0,0].item():.3f}")
        
        # Normalize content to zero mean, unit variance
        normalized = (content_feat - c_mean) / (c_std + 1e-5)  # 1e-5 prevents division by zero
        
        # Re-scale to match style statistics
        return normalized * s_std + s_mean

class Decoder(nn.Module):
    """
    Decoder network that upsamples fused features back to ab color channels.
    
    Architecture mirrors VGG's downsampling but in reverse:
    - VGG downsamples images through pooling/striding
    - Decoder upsamples through nearest neighbor interpolation
    - Gradually reduces channel count while increasing spatial resolution
    
    Final output: 2 channels representing a* and b* in Lab color space
    """
    
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # Start with 512 channels from AdaIN-fused features
            nn.Conv2d(512, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Double spatial resolution
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Final layer: output 2 channels (a* and b* color channels)
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()  # Constrain outputs to [-1,1] range (matches our Lab normalization)
        )
        
    def forward(self, x):
        """
        Decode fused features to color channels.
        
        Args:
            x (torch.Tensor): Fused features from AdaIN (N, 512, H/16, W/16)
            
        Returns:
            torch.Tensor: Predicted ab channels (N, 2, H, W) in range [-1,1]
        """
        return self.model(x)

class StyleColorizer(nn.Module):
    """
    Complete style-transfer colorization network.
    
    Pipeline:
    1. LIFT: Convert single-channel grayscale to rich feature representation
    2. EXTRACT: Get style features from reference image via VGG
    3. FUSE: Apply AdaIN to transfer style statistics to content
    4. DECODE: Generate final a* and b* color channels
    
    This architecture separates luminance (L) from chrominance (a*,b*),
    allowing the model to preserve brightness information while predicting colors.
    """
    
    def __init__(self, content_layers=['21'], style_layers=['0','5','10','19','28']):
        """
        Initialize the complete colorization network.
        
        Args:
            content_layers (list): VGG layer indices for content representation
            style_layers (list): VGG layer indices for style representation
        """
        super().__init__()
        
        # Feature extractor for computing style/content losses and AdaIN
        self.feature_extractor = VGGFeatureExtractor(content_layers, style_layers)
        
        # Style transfer mechanism
        self.adain = AdaIN()
        
        # Upsampling decoder
        self.decoder = Decoder()
        
        # Lifting network: transforms single L channel into rich 512-channel representation
        # This gives the model capacity to understand the grayscale content before style fusion
        self.lift = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
            # Output: (N, 512, H, W) - rich content representation
        )
        
    def forward(self, L_tensor, style_rgb_norm):
        """
        Complete forward pass: grayscale + style → colorized ab channels.
        
        Args:
            L_tensor (torch.Tensor): Grayscale input (N,1,H,W) in [0,1]
            style_rgb_norm (torch.Tensor): Style image (N,3,H,W) ImageNet-normalized
            
        Returns:
            ab_pred (torch.Tensor): Predicted ab channels (N,2,H,W) in [-1,1]
        """
        # Step 1: LIFT grayscale to rich feature representation
        content_feat = self.lift(L_tensor)  # (N,1,H,W) → (N,512,H,W)
        
        # Step 2: EXTRACT style features from reference image
        style_features = self.feature_extractor(style_rgb_norm)
        style_content = style_features['content']  # High-level style representation for AdaIN
        
        # Step 3: FUSE content and style via AdaIN
        # Transfer style's mean/variance to content features
        fused = self.adain(content_feat, style_content)  # (N,512,H,W) with style statistics
        
        # Step 4: DECODE fused features to color channels
        ab_pred = self.decoder(fused)  # (N,512,H,W) → (N,2,H,W)
        
        return ab_pred

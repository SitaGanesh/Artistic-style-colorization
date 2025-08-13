"""
Data utilities for Task 1: Artistic Style Transfer + Colorization
Handles image loading, preprocessing, and color space conversions.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Standard ImageNet normalization (for pretrained VGG features)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#Load image->check for RGB format->build tensor for image->create batch dimentions-> move tensors to cpu
def load_image(image_path, target_size=(256, 256), device='cpu'):
    """
    Load an image from disk, resize it, and convert to a PyTorch tensor.

    1. Verify the file exists, else raise FileNotFoundError.
    2. Use PIL to open and ensure the image is in RGB format.
    3. Build a torchvision transform to:
       a. Resize the image to target_size (H, W).
       b. Convert the image to a FloatTensor in [0,1], shape (C, H, W).
    4. Add a batch dimension: (1, 3, H, W).
    5. Move the tensor to the specified device ('cpu' or 'cuda').

    Args:
        image_path (str): Path to the image file on disk.
        target_size (tuple of int): Desired (height, width) for resizing.
        device (str): Device specifier for the returned tensor.

    Returns:
        torch.Tensor: A 4D tensor of shape (1, 3, H, W) on the given device.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and convert to RGB (Python Image Library)
    image = Image.open(image_path).convert('RGB')
    
    # Create transform pipeline(torchvision transform used for Data processing and data argumentation in computer vision task)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Converts to [0,1] and changes to (C,H,W)
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, H, W)
    
    return tensor.to(device)

#convert tensors into numpy->give to OpenCV-> converts rgb to bgr->convert bgr to lab-> exract the L channels and normailze->convert back to rgb tensors
def load_grayscale_image(image_path, target_size=(256, 256), device='cpu'):
    """
    Load an image, convert to Lab color space, and return only the luminance (L) channel.

    This function is crucial for colorization because:
    1. We need the L channel (brightness/luminance) as input to our model
    2. The model will predict the a* and b* channels (color information)
    3. Lab color space separates brightness from color, making it ideal for colorization

    Process:
    1. Load RGB image using our load_image function
    2. Convert PyTorch tensor to numpy array for OpenCV processing
    3. Scale from [0,1] to [0,255] for OpenCV compatibility
    4. Convert RGB→BGR (OpenCV uses BGR format internally)
    5. Convert BGR→Lab using OpenCV's color space conversion
    6. Extract only the L channel and normalize from [0,100] to [0,1]
    7. Convert back to PyTorch tensor with proper dimensions

    Args:
        image_path (str): Path to the image file on disk
        target_size (tuple): Desired (height, width) for resizing
        device (str): Device specifier for the returned tensor

    Returns:
        torch.Tensor: Luminance tensor of shape (1, 1, H, W) in range [0,1]
    """
    # Load RGB image first
    rgb_tensor = load_image(image_path, target_size, device='cpu')
    
    # Convert RGB to Lab color space
    rgb_np = rgb_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)
    rgb_np = (rgb_np * 255).astype(np.uint8)
    
    # OpenCV uses BGR, so convert RGB->BGR->Lab
    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    lab_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2LAB)
    
    # Extract L channel (luminance) and normalize to [0,1]
    L_channel = lab_np[:, :, 0] / 100.0  # Lab L is in [0,100]
    
    # Convert back to tensor format
    L_tensor = torch.from_numpy(L_channel).float().unsqueeze(0).unsqueeze(0)
    
    return L_tensor.to(device)

#convert rgb tensors to lab space normailze->opencv bgr->normailize->pytorch tensors
def rgb_to_lab_tensor(rgb_tensor):
    """
    Convert an RGB tensor to normalized Lab space.

    This is used when computing style/content losses directly in Lab
    or when you need Lab representation for analysis.

    Args:
        rgb_tensor (torch.Tensor): Input RGB tensor of shape (1, 3, H, W),
                                   with values in [0,1].

    Returns:
        torch.Tensor: Output Lab tensor of shape (1, 3, H, W):
                      - L channel normalized to [0,1] (original [0,100])
                      - a and b channels normalized to [-1,1] (original [0,255] centered at 128)
    """
    # 1) Move to CPU and convert to numpy array
    #    Remove batch dim (1,3,H,W) -> (3,H,W), then permute to (H,W,3)
    rgb_np = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #    Scale from [0,1] to [0,255] uint8 for OpenCV
    rgb_np = (rgb_np * 255).astype(np.uint8)
    
    # 2) Convert color spaces via OpenCV
    #    OpenCV uses BGR internally, so convert RGB->BGR first
    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    #    Then convert BGR->Lab (output float32)
    lab_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 3) Normalize Lab channels for network-friendly ranges
    #    L channel: originally [0,100] -> normalize to [0,1]
    lab_np[:, :, 0] = lab_np[:, :, 0] / 100.0
    #    a, b channels: originally [0,255] centered at 128
    #    subtract 128 then divide by 128 to map to [-1,1]
    lab_np[:, :, 1] = (lab_np[:, :, 1] - 128.0) / 128.0
    lab_np[:, :, 2] = (lab_np[:, :, 2] - 128.0) / 128.0
    
    # 4) Convert back to PyTorch tensor with shape (1, 3, H, W)
    lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).unsqueeze(0)
    
    return lab_tensor

def lab_to_rgb_tensor(lab_tensor):
    """
    Convert a normalized Lab tensor back to an RGB tensor.

    This reverses `rgb_to_lab_tensor`, denormalizing and converting color 
    space so we can visualize or save the final colorized output.

    Steps:
    1. Detach from computation graph and move to CPU numpy array.
    2. Denormalize channels:
       - L channel scaled from [0,1] → [0,100]
       - a and b channels scaled from [-1,1] → [1,255]
    3. Clip each channel to valid OpenCV Lab ranges.
    4. Convert the Lab image (uint8) directly to RGB with cv2.COLOR_LAB2RGB.
    5. Convert the resulting RGB numpy back into a torch tensor in [0,1].

    Args:
        lab_tensor (torch.Tensor): Lab tensor of shape (1, 3, H, W),
                                   where L is [0,1], a/b are [-1,1].

    Returns:
        torch.Tensor: RGB tensor of shape (1, 3, H, W) with values in [0,1].
    """
    import numpy as np
    import cv2

    # 1) Prepare numpy array
    #    (1,3,H,W) → (H,W,3), detach gradients, convert to float32
    lab_np = lab_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    # 2) Denormalize each channel
    lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0            # L: [0,1] → [0,100]
    lab_np[:, :, 1] = (lab_np[:, :, 1] * 127.0) + 128.0  # a: [-1,1] → [1,255]
    lab_np[:, :, 2] = (lab_np[:, :, 2] * 127.0) + 128.0  # b: [-1,1] → [1,255]

    # 3) Clip to ensure valid Lab ranges for OpenCV
    lab_np[:, :, 0] = np.clip(lab_np[:, :, 0], 0, 100)
    lab_np[:, :, 1] = np.clip(lab_np[:, :, 1], 1, 255)
    lab_np[:, :, 2] = np.clip(lab_np[:, :, 2], 1, 255)

    # 4) Convert to uint8 and apply OpenCV’s Lab→RGB
    lab_uint8 = lab_np.astype(np.uint8)
    rgb_np = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)

    # 5) Convert numpy RGB [0,255] → tensor [0,1]
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    return rgb_tensor

def normalize_for_vgg(tensor):#VGG is a Visual Geomentry Group contains large number of layers in a network called Deep. it is widely used in Deep Convolutional Neural Network
    """
    Normalization is a data preparation technique that scales numerical features to similar range -1 to 1 or 0 to 1
    
    Normalize an RGB tensor using ImageNet mean and std for VGG input.

    Pretrained VGG-19 expects inputs zero-centered and scaled according to
    ImageNet statistics. This function applies that normalization.

    Args:
        tensor (torch.Tensor): Input RGB tensor of shape (1, 3, H, W),
                               with values in [0,1].

    Returns:
        torch.Tensor: Normalized tensor of the same shape, ready for VGG.
    """
    # 1) Create mean and std tensors shaped (1,3,1,1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    
    # 2) Move mean/std to GPU if input is on CUDA
    if tensor.is_cuda:
        mean = mean.cuda()
        std  = std.cuda()
    
    # 3) Apply normalization: (x - mean) / std
    return (tensor - mean) / std

def denormalize_from_vgg(tensor):
    """
    Reverse VGG ImageNet normalization to recover [0,1] RGB values.

    This undoes `normalize_for_vgg`, converting a tensor that was
    zero-centered and scaled by ImageNet mean/std back to its original
    [0,1] range so it can be visualized or saved.

    Args:
        tensor (torch.Tensor): Normalized tensor of shape (1, 3, H, W),
                               which was normalized via (x - mean)/std.

    Returns:
        torch.Tensor: Denormalized tensor in [0,1], same shape.
    """
    # 1) Create mean and std tensors shaped (1,3,1,1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    
    # 2) If input is on GPU, move mean/std there too
    if tensor.is_cuda:
        mean = mean.cuda()
        std  = std.cuda()
    
    # 3) Reverse normalization: x * std + mean
    return (tensor * std) + mean


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL Image for easy display or saving.

    Args:
        tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W)
                               with values in [0,1].

    Returns:
        PIL.Image: Output image in RGB mode with pixel values [0,255].
    """
    # 1) Remove the batch dimension: (1,3,H,W) -> (3,H,W)
    img = tensor.squeeze(0)
    # 2) Move channels to last dimension: (3,H,W) -> (H,W,3)
    img = img.permute(1, 2, 0).cpu().numpy()
    
    # 3) Clamp values to [0,1] then scale to [0,255]
    img = np.clip(img, 0, 1) * 255
    # 4) Convert to uint8 for PIL compatibility
    img = img.astype(np.uint8)
    
    # 5) Create and return a PIL Image (mode 'RGB')
    return Image.fromarray(img)


def get_image_paths(folder_path, extensions=('.jpg', '.jpeg', '.png')):
    """
    Retrieve sorted image file paths from a directory.

    This helper scans the specified folder for files matching the
    given extensions and returns a sorted list of full file paths.

    Args:
        folder_path (str): Path to the directory containing images.
        extensions (tuple of str): File extensions to include (case-insensitive).

    Returns:
        list of str: Sorted full paths to matching image files.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
    """
    # 1) Validate the directory exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # 2) Collect files with matching extensions
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            full_path = os.path.join(folder_path, filename)
            image_paths.append(full_path)
    
    # 3) Return the list sorted alphabetically
    return sorted(image_paths)

def get_lab_mean_std(style_path):
    """
    Compute the mean and standard deviation of the a* and b* channels
    in Lab color space for a given style image. This allows us to
    directly match the style image’s global color palette.

    Steps:
    1. Load the image using OpenCV (BGR format).
    2. Convert BGR to Lab color space (float32).
    3. Split into L, a, b channels.
    4. Normalize the a and b channels from [0,255] centered at 128
       to the model’s expected [-1,1] range.
    5. Compute and return the mean and standard deviation of the
       normalized a and b channels.

    Args:
        style_path (str): File path to the style image.

    Returns:
        tuple of float:
            a_mean (float): Mean of normalized a channel.
            a_std  (float): Std deviation of normalized a channel.
            b_mean (float): Mean of normalized b channel.
            b_std  (float): Std deviation of normalized b channel.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    import cv2
    import numpy as np

    # 1) Load image in BGR
    img_bgr = cv2.imread(style_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Style image not found: {style_path}")

    # 2) Convert to Lab (float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 3) Split into channels
    L_channel, a_channel, b_channel = cv2.split(lab)

    # 4) Normalize a,b channels to [-1,1]
    #    Original range: [0,255], center 128
    a_norm = (a_channel - 128.0) / 127.0
    b_norm = (b_channel - 128.0) / 127.0

    # 5) Compute statistics
    return a_norm.mean(), a_norm.std(), b_norm.mean(), b_norm.std()

# Test function to verify everything works
def test_data_utils():
    """
    Run a comprehensive check of all data utility functions to ensure:
      1. Content and style directories exist and contain images.
      2. load_image correctly loads and tensors an RGB image.
      3. load_grayscale_image correctly extracts the L channel.
      4. rgb_to_lab_tensor maps RGB to Lab properly.
      5. lab_to_rgb_tensor maps Lab back to RGB properly.
    
    Usage:
        Run this from a notebook or script: `test_data_utils()`
        It will print status messages and shapes for each step.
    """
    print("Testing data utilities...")
    
    try:
        # 1) Verify content and style folders exist
        content_dir = "../data/content"
        style_dir   = "../data/styles"
        
        if os.path.exists(content_dir):
            content_files = get_image_paths(content_dir)
            print(f"Found {len(content_files)} content images:")
            # List first 3 filenames
            for f in content_files[:3]:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"Content directory not found: {content_dir}")
            return
            
        if os.path.exists(style_dir):
            style_files = get_image_paths(style_dir)
            print(f"Found {len(style_files)} style images:")
            # List all style filenames
            for f in style_files:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"Style directory not found: {style_dir}")
            return
        
        # 2) If we have at least one content and one style image, test conversions
        if content_files and style_files:
            content_path = content_files[0]
            style_path   = style_files[0]
            
            print("\nTesting with:")
            print(f"  Content: {os.path.basename(content_path)}")
            print(f"  Style:   {os.path.basename(style_path)}")
            
            # a) load_image: should return a (1,3,H,W) tensor
            rgb_img = load_image(content_path)
            print(f"RGB image shape:       {rgb_img.shape}")
            
            # b) load_grayscale_image: should return a (1,1,H,W) tensor
            gray_img = load_grayscale_image(content_path)
            print(f"Grayscale image shape: {gray_img.shape}")
            
            # c) rgb_to_lab_tensor: should return a (1,3,H,W) Lab tensor
            lab_img = rgb_to_lab_tensor(rgb_img)
            print(f"Lab image shape:       {lab_img.shape}")
            
            # d) lab_to_rgb_tensor: should revert back to (1,3,H,W) RGB
            rgb_back = lab_to_rgb_tensor(lab_img)
            print(f"RGB back shape:        {rgb_back.shape}")
            
            print("All tests passed!")
        else:
            print("Please add sample images to data/content/ and data/styles/")
            
    except Exception as e:
        # If any step fails, print the error and traceback
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_utils()

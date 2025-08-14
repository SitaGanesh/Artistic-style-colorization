"""
Streamlit GUI for Task 1: Artistic Style Transfer + Colorization
Interactive web interface for colorizing grayscale images with artistic styles.
"""
import streamlit as st
import torch
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path for root-level execution
sys.path.append('./task1_style_transfer/src')

from data_utils import (
    load_grayscale_image,
    load_image, 
    normalize_for_vgg,
    lab_to_rgb_tensor,
    tensor_to_pil
)
from model import StyleColorizer

# Page configuration
st.set_page_config(
    page_title="Artistic Colorization",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleColorizer().to(device)
    
    # Updated path for root directory
    model_path = './task1_style_transfer/models/baseline_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None, device

def get_style_images():
    """Get available style images"""
    # Updated path for root directory
    style_dir = './task1_style_transfer/data/styles'
    if not os.path.exists(style_dir):
        return []
    
    style_files = []
    for file in os.listdir(style_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            style_files.append(os.path.join(style_dir, file))
    
    return sorted(style_files)

def colorize_image(model, device, grayscale_img, style_img, style_strength=1.0, content_weight=1.0, color_saturation=1.0):
    """Colorize a grayscale image with the given style"""
    try:
        # Convert PIL images to tensors
        L_tensor = load_grayscale_image_from_pil(grayscale_img, device=device)
        style_tensor = load_image_from_pil(style_img, device=device)
        style_norm = normalize_for_vgg(style_tensor)
        
        with torch.no_grad():
            ab_pred = model(L_tensor, style_norm)
            
            # Handle shape mismatch
            _, _, H, W = L_tensor.shape
            if ab_pred.shape[2:] != (H, W):
                ab_pred = torch.nn.functional.interpolate(
                    ab_pred, size=(H, W), mode='bilinear', align_corners=False
                )
            
            # Apply style strength and color saturation
            ab_pred = ab_pred * style_strength * color_saturation
            
            # Direct color transfer from style
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                style_img.save(tmp.name)
                style_path = tmp.name
            
            try:
                from data_utils import get_lab_mean_std
                a_mean, a_std, b_mean, b_std = get_lab_mean_std(style_path)
                
                # Re-scale ab channels
                a = ab_pred[0,0]
                b = ab_pred[0,1]
                
                a = (a - a.mean()) / (a.std() + 1e-5) * a_std + a_mean
                b = (b - b.mean()) / (b.std() + 1e-5) * b_std + b_mean
                
                ab_pred = torch.stack([a, b], dim=0).unsqueeze(0)
            finally:
                os.unlink(style_path)
            
            # Reconstruct RGB
            lab_full = torch.cat([L_tensor, ab_pred], dim=1)
            rgb_pred = lab_to_rgb_tensor(lab_full)
            
        return tensor_to_pil(rgb_pred)
    
    except Exception as e:
        st.error(f"Error during colorization: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def load_grayscale_image_from_pil(pil_img, target_size=(256, 256), device='cpu'):
    """Load PIL image and convert to grayscale tensor"""
    # Convert to grayscale if not already
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    
    # Resize
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(pil_img) / 255.0  # Normalize to [0,1]
    L_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
    
    return L_tensor.to(device)

def load_image_from_pil(pil_img, target_size=(256, 256), device='cpu'):
    """Load PIL image and convert to RGB tensor"""
    # Convert to RGB
    pil_img = pil_img.convert('RGB').resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(pil_img) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor.to(device)

# Main App
def main():
    st.title("🎨 Artistic Style-Based Colorization")
    st.markdown("Transform your grayscale photos with artistic styles!")
    
    # Load model
    model, device = load_model()
    if model is None:
        st.stop()
    
    # Get available styles
    style_files = get_style_images()
    if not style_files:
        st.error("No style images found in ./task1_style_transfer/data/styles/. Please add some style images first.")
        st.stop()
    
    with st.container():
        st.header("📤 Upload & Configure")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a grayscale image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a grayscale or color image (will be converted to grayscale)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", width=300)
            
            # Style selection
            st.subheader("🎭 Choose Artistic Style") 
            
            # Show style thumbnails in a grid
            if style_files:
                cols_per_row = 4
                rows = (len(style_files) + cols_per_row - 1) // cols_per_row
                
                selected_style_idx = st.session_state.get('selected_style_idx', 0)
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        style_idx = row * cols_per_row + col_idx
                        if style_idx < len(style_files):
                            with cols[col_idx]:
                                try:
                                    style_file = style_files[style_idx]  
                                    style_name = os.path.basename(style_file).replace('_', ' ').title()
                                    
                                    # Show thumbnail
                                    style_img = Image.open(style_file)
                                    st.image(style_img, caption=style_name, width=120)
                                    
                                    # Selection button
                                    if st.button(f"Select", key=f"style_{style_idx}"):
                                        st.session_state['selected_style_idx'] = style_idx
                                        selected_style_idx = style_idx
                                        
                                except Exception as e:
                                    st.error(f"Could not load {style_name}")
                
                # Also keep dropdown as backup
                style_names = [os.path.basename(f).replace('_', ' ').title() for f in style_files]
                selected_name = st.selectbox(
                    "Or select from dropdown:",
                    style_names,
                    index=selected_style_idx,
                    key="style_dropdown"
                )
                
                # Update index if dropdown changed
                if selected_name != style_names[selected_style_idx]:
                    selected_style_idx = style_names.index(selected_name)
                    st.session_state['selected_style_idx'] = selected_style_idx

        # Real-time preview section
        if uploaded_file and 'selected_style_idx' in st.session_state:
            st.subheader("🔄 Real-Time Preview")
            
            # Parameter controls with live update
            col1, col2 = st.columns(2)
            
            with col1:
                style_strength = st.slider(
                    "Style Strength", 0.1, 2.0, 1.0, 0.1, key="live_style"
                )
                content_weight = st.slider(
                    "Content Preservation", 0.1, 2.0, 1.0, 0.1, key="live_content"
                )
            
            with col2:
                color_saturation = st.slider(
                    "Color Saturation", 0.5, 2.0, 1.0, 0.1, key="live_saturation"
                )
                enable_preview = st.checkbox("Enable Live Preview", value=False)
            
         
            # Live preview logic
            if enable_preview or st.button("Generate Preview"):
                preview_size = (128, 128)
                
                with st.spinner("Generating preview..."):
                    try:
                        style_img = Image.open(style_files[st.session_state['selected_style_idx']])
                        
                        # FIXED: Correct parameter passing
                        preview_result = colorize_image(
                            model, device, 
                            input_image.resize(preview_size), 
                            style_img.resize(preview_size), 
                            style_strength, content_weight, color_saturation  # Now matches function signature
                        )
                        
                        if preview_result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.image(input_image.resize(preview_size).convert('L'), 
                                        caption="Input (Preview)", width=120)
                            with col2:
                                st.image(style_img.resize(preview_size), 
                                        caption="Style (Preview)", width=120)
                            with col3:
                                st.image(preview_result, 
                                        caption="Result (Preview)", width=120)
                                
                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")
                        import traceback
                        st.error(f"Full error: {traceback.format_exc()}")
            
            # Colorize button
            if st.button("🎨 Colorize Image", type="primary"):
                with st.spinner("Applying artistic colorization..."):
                    try:
                        # Load selected style
                        style_img = Image.open(style_files[selected_style_idx])
                        
                        # Perform colorization
                        result = colorize_image(
                            model, device, input_image, style_img, style_strength
                        )
                        
                        if result is not None:
                            # Store result in session state
                            st.session_state['colorized_result'] = result
                            st.session_state['input_image'] = input_image
                            st.session_state['style_image'] = style_img
                            st.success("Colorization completed!")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    
        st.header("🖼️ Results")
        
        if 'colorized_result' in st.session_state:
            # Show before/after comparison
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.subheader("Input")
                input_gray = st.session_state['input_image'].convert('L')
                st.image(input_gray, caption="Grayscale Input")
            
            with result_col2:
                st.subheader("Style")
                st.image(st.session_state['style_image'], caption="Style Reference")
            
            with result_col3:
                st.subheader("Result")
                st.image(st.session_state['colorized_result'], caption="Colorized Output")
            
            # Download button
            st.markdown("---")
            
            # Convert PIL image to bytes for download
            import io
            buffer = io.BytesIO()
            st.session_state['colorized_result'].save(buffer, format="PNG")
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Result",
                data=buffer.getvalue(),
                file_name="colorized_image.png",
                mime="image/png"
            )
        
        else:
            st.info("Upload an image and click 'Colorize Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ by Sita Ganesh using Streamlit and PyTorch | "
        f"Device: {device} | "
        f"Styles available: {len(style_files)}"
    )

if __name__ == "__main__":
    main()

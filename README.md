# Artistic Style Transfer \& Colorization Project

**Transform grayscale images into vibrant artworks by applying the color palettes and textures of famous paintings.**
This README guides you through every component—from setup to deployment—so that beginners and experts alike can understand, run, and extend the project.


## Project Overview

This project implements a **StyleColorizer** model that:

- **Preserves image structure** using the L channel of Lab color space
- **Transfers artistic style** via Adaptive Instance Normalization (AdaIN)
- **Generates smooth, vibrant colors** with a decoder network

By leveraging a pretrained VGG-19 feature extractor and custom loss functions, our model learns to colorize any grayscale image in the style of Monet, Van Gogh, Picasso, and more.

***

## Folder Structure

```
.
├── data/
│   ├── content/         # Grayscale input images (png, jpg)
│   └── styles/          # Style reference images (jpg, png)
│
├── models/              # Saved model checkpoints (.pth)
│
├── results/             # Generated outputs & montages
│
├── src/                 # Source code
│   ├── data_utils.py    # Image I/O & Lab/RGB conversion
│   ├── model.py         # StyleColorizer architecture (VGGFeatureExtractor, AdaIN, Decoder)
│   ├── losses.py        # Content, style, and TV loss functions
│   ├── train.py         # Command-line training script
│   ├── inference.py     # Single-image inference script
│   └── evaluation.py    # Evaluation metrics & ModelEvaluator class
│
├── notebooks/           # Jupyter notebooks
│   ├── baseline.ipynb   # Mini training loop & sanity checks
│   └── evaluation.ipynb # Interactive evaluation & visualization
│
├── app.py               # Flask/Streamlit GUI application
│
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore
```


***

## Requirements

1. **requirements.txt**

```txt
torch>=2.0
torchvision
numpy
opencv-python
pillow
matplotlib
scikit-image
Streamlit
```

2. **Model Training Notebook (.ipynb)**
    - `notebooks/baseline.ipynb`
    - Demonstrates data loading, forward pass, mini training loop, and saving results.
3. **Model Weights**
    - Baseline trained checkpoint: [`models/baseline_model.pth`]
4. **GUI**
    - `app.py` uses Streamlit to provide a web interface for uploading images and viewing results.

5. **Evaluation Metrics**
    - Unlike classification, colorization is a regression task. We use:
        - **PSNR** (Peak Signal-to-Noise Ratio)
        - **SSIM** (Structural Similarity Index)
        - **Content Fidelity** (SSIM on L channel)
        - **Style Effectiveness** (Gram-MSE on VGG features)
        - **Color Diversity** (std. dev. of ab channels)
    - _Confusion matrix, precision, and recall do **not** apply here._
6. **Accuracy Requirement**
    - For regression tasks, “70% accuracy” is undefined. Instead, target:
        - **SSIM > 0.7** on ground-truth-paired datasets
        - **PSNR > 20 dB**

***

## Installation \& Local Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/colorization_task1.git
cd colorization_task1
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows(command prompt)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify installation by running:

```bash
python src/train.py --help
python src/inference.py --help
```


***

## Data Preparation

- **Download sample images** into `data/content/` (grayscale) and `data/styles/` (RGB color).
- Recommended: at least **100 content images** and **10 style images** for meaningful training.
- Ensure images are **256×256** or will be resized.

***

## Model Training

1. **Quick Start** (single image mini-loop):

```bash
jupyter notebook notebooks/baseline.ipynb
```

2. **Full Training**:

```bash
python src/train.py \
  --data_dir data \
  --styles_dir data/styles \
  --output_dir models \
  --epochs 10 \
  --batch_size 4 \
  --lr 1e-4
```

3. **Checkpointing**
    - Models saved each epoch in `models/`.

***

## Pretrained Weights \& Checkpoints

- **Baseline (mini-loop)**: `models/baseline_model.pth`
- **Losses**:
  
  ![task1i2](https://github.com/user-attachments/assets/d94ea1d6-f21d-4589-bf54-c827dc68b5d0)


***

## Evaluation

### Automated Metrics

```bash
python -c "from src.evaluation import ModelEvaluator; \
import torch; \
from src.model import StyleColorizer, VGGFeatureExtractor; \
m=StyleColorizer(); \
m.load_state_dict(torch.load('models/baseline_model.pth')); \
ve=VGGFeatureExtractor(['21'], ['0','5','10','19','28']); \
# build a DataLoader for test set…
# metrics = ModelEvaluator(m, ve).evaluate_dataset(test_loader, style_tensors)
# ModelEvaluator(m, ve).print_summary(metrics)
"
```
![task1i1](https://github.com/user-attachments/assets/0dbda3aa-753e-4c35-b7eb-0e67bb6a977d)


### Notebook

```bash
jupyter notebook notebooks/evaluation.ipynb
```

**Key Results to Aim For**

- SSIM > 0.7 on held-out test pairs
- PSNR > 20 dB
- Content Fidelity > 0.5
- Style Effectiveness > 0.9
- Color Diversity in

***


### Web GUI

1. Install Streamlit:

```bash
pip install Streamlit
```

2. Run the app:

```bash
cd task1_style_transfer/gui/

cd colorization_task1\task1_style_transfer\gui>streamlit run app.py
```

3. Open `http://localhost:8501/` in your browser
4. Upload grayscale \& style images, click **Colorize**, download result

_**Space for GUI screenshots:**_

***

## Hardware \& Resource Requirements

- **CPU-only**: works but slow (processing ~1 image/sec)
- **GPU recommended**: NVIDIA GPU with ≥4 GB VRAM for faster training (4–8 images/sec)
- **Storage**:
    - Source code \& notebooks: ~100 MB
    - Data: ~500 MB for 1,000 images
    - Models: ~200 MB total
- **Memory**: ≥8 GB RAM for training with batch size 4

***

## Integration \& Grading Criteria


**Evaluation**:

- **Model Performance** (continuous metrics): SSIM, PSNR, Content/Style Fidelity
- **Code Quality**: Modular design, documentation, clear variable names
- **Reproducibility**: `requirements.txt`, fixed random seeds, checkpointing
- **User Experience**: Clear README, working CLI, intuitive GUI
- **Bonus**: Real-time inference, support for arbitrary image sizes, additional styles

***

# Output

![task1i3](https://github.com/user-attachments/assets/bf4325e1-ca13-4c25-917a-1bfbffc96779)
![task1i4](https://github.com/user-attachments/assets/e8439044-b320-4260-9ef9-2657ebd760c6)
![task1i5](https://github.com/user-attachments/assets/7f725724-1322-4fa8-a32c-6755ee0bd1cf)
![task1i6](https://github.com/user-attachments/assets/08f2d49e-511d-4add-b494-618d28ff812c)

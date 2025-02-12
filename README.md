# Monocular Depth Estimation

## Overview
This repository contains the implementation of a lightweight and efficient neural network model for monocular depth estimation from sequential image streams. The model aims to provide accurate depth predictions while optimizing computational efficiency for real-time applications in robotics and embedded systems. It leverages MobileNetV2 for feature extraction and a custom decoder for depth estimation.

## Features
- **Lightweight Architecture**: Uses MobileNetV2 as the encoder for efficient feature extraction.
- **Optimized Depth Decoding**: Custom depth decoder balances accuracy and efficiency.
- **Self-Supervised Learning**: Utilizes reprojection loss for training without ground-truth depth labels.
- **Real-Time Performance**: Achieves high frame rates suitable for embedded systems.
- **Post-Training Quantization**: Reduces model size and inference latency.

## Dataset
The model is trained and evaluated on the **KITTI dataset**, a widely used benchmark for depth estimation. The dataset consists of real-world driving scenarios with high-resolution RGB images and corresponding sparse depth maps.

## Model Architecture
### 1. Encoder: MobileNetV2
- Pretrained on ImageNet
- Extracts multi-scale features at different resolutions
- Optimized for speed and low memory usage

### 2. Decoder: DepthDecoder
- Upsamples and refines depth maps using hierarchical feature fusion
- Incorporates nearest-neighbor interpolation and convolution layers
- Uses sigmoid activation for final depth predictions

## Loss Function
The model is trained using a combination of:
- **Reprojection Loss** (SSIM + L1 Loss) for self-supervised learning
- **Smoothness Loss** to enforce spatial consistency in depth predictions

## Installation
```bash
# Clone the repository
git clone https://github.com/akoushik2k/Mono-Depth-Estimation.git
cd Mono-Depth-Estimation

# Install dependencies
pip install -r requirements.txt
```

## Training
```bash
python train.py --dataset_path /path/to/kitti --epochs 20 --batch_size 8
```

## Evaluation
```bash
python evaluate.py --dataset_path /path/to/kitti --checkpoint best_model.pth
```

## Results
| Model      | AbsRel | RMSE  | FPS  |
|------------|--------|-------|------|
| Ours       | 0.32   | 4.834 | 258.4|
| Monodepth2 | 0.106  | 4.750 | 273.3|
| SPIdepth   | 0.029  | 1.394 | 642.4|

## Author Contributions
- **Kiran Kommaraju** - Model architecture and optimization
- **Koushik Alapati** - Dataset preparation and preprocessing
- **Sai Dinesh Gelam** - Training and hyperparameter tuning
- **Raghu Dharahas Reddy Kotla** - Comparative analysis and evaluation

## Future Work
- Improve accuracy on high-resolution inputs
- Further optimize for embedded devices
- Explore alternative lightweight architectures

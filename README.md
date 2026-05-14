## install
```bash
mkdir torch-tutorials
cd torch-tutorials

uv init --python 3.12
uv venv

source .venv/bin/activate 
# windows: .venv\Scripts\Activate.ps1

uv pip install torch=="2.2.0" torchvision torchaudio
uv add "numpy<2"

uv run python -c 'import torch; print(f"PyTorch version: {torch.__version__}")'
```

## Introduction to PyTorch
 - What is PyTorch?
    PyTorch is an open-source deep learning framework developed by Meta AI. It is widely used for:

- Machine Learning
- Deep Learning
- Computer Vision
- Natural Language Processing
- Research and Production AI Systems

### PyTorch is popular because it is:
 - Easy to learn
- Pythonic
- Flexible
- Fast
- Great for experimentation

## Table of Contents
- Installing PyTorch
- Understanding Tensors
- Tensor Operations
- Working with GPU
- Autograd and Gradients
- Building Neural Networks
- Activation Functions
- Loss Functions
- Optimizers
- Training Loop
- Dataset and DataLoader
- Image Classification Project
- Saving and Loading Models
- Model Evaluation
- Tips and Best Practices
- Final Project Ideas
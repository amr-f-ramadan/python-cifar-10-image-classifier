# CIFAR-10 Image Classifier - Udacity Machine Learning Nanodegree Project

**⚠️ This Project was first submitted on September 15, 2024. Later on, it went through some enhancements that are listed below.**

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-yellow.svg)

## 🎯 Project Overview

This project implements a deep learning solution for the CIFAR-10 image classification challenge as part of the **Udacity Introduction to Machine Learning with PyTorch Nanodegree**. The solution features two distinct CNN architectures designed to achieve high accuracy on the CIFAR-10 dataset while maintaining computational efficiency.

### 🏆 Key Achievements
- ✅ **Successfully exceeded 70% accuracy requirement** (beating Detectocorp's benchmark)
- 🚀 **Implemented two distinct architectures**: Simple CNN and Residual CNN
- ⚡ **Cross-platform GPU acceleration** support (CUDA, MPS, CPU)
- 📊 **Comprehensive training visualization** and model evaluation
- 🔧 **Production-ready environment setup** with automated conda configuration

## 📊 Performance Results

| Model | Architecture | Validation Accuracy | Test Accuracy | Parameters |
|-------|-------------|-------------------|---------------|------------|
| Simple CNN | 3-layer CNN + FC | 70%+ | 68%+ | ~2M |
| Residual CNN | ResNet-inspired | 80%+ | 78%+ | ~11M |

## 🏗️ Architecture Details

### Simple CNN Model
- **Convolutional Layers**: 3 layers (16→32→64 filters)
- **Pooling**: MaxPool2d after each conv layer
- **Fully Connected**: 2048→1024→10 neurons
- **Regularization**: Dropout (0.5)
- **Activation**: ReLU + LogSoftmax

### Residual CNN Model
- **Residual Blocks**: 4 layers with skip connections
- **Batch Normalization**: After each convolution
- **Adaptive Pooling**: Global average pooling
- **Channels**: 64→128→256→512 progression
- **Advanced Architecture**: Inspired by ResNet

## 🚀 Quick Start

### Prerequisites
- **Anaconda/Miniconda** installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- **macOS/Linux/Windows** with Python 3.9+
- **GPU (optional)**: NVIDIA GPU with CUDA or Apple Silicon with MPS

### Setup & Run (One Command)

```bash
# Make setup script executable and run it
chmod +x setup_environment.sh && ./setup_environment.sh

# Activate environment and launch notebook
source ./activate_env.sh
jupyter notebook CIFAR-10_Image_Classifier-STARTER_Solution.ipynb
```

### What the Setup Script Does
- 🔍 **Auto-detects** your operating system and hardware
- 📦 **Creates** a dedicated conda environment (`cifar10-classifier`)
- 🔥 **Installs PyTorch** with optimal GPU support:
  - **macOS**: MPS acceleration for Apple Silicon
  - **Linux**: CUDA support for NVIDIA GPUs
  - **Windows/Others**: CPU optimized version
- 📚 **Installs** all required packages (matplotlib, numpy, jupyter, tqdm)
- 🔗 **Sets up** Jupyter kernel for the environment
- ✅ **Tests** the installation automatically
- 📝 **Creates** activation script for easy future use
- 💾 **Exports** environment.yml for reproducibility

### Manual Activation (Future Use)
```bash
# Quick activation
source ./activate_env.sh

# Or manual conda activation
conda activate cifar10-classifier
```

## 🎯 Project Structure

```
python-cifar-10-image-classifier/
├── 📓 CIFAR-10_Image_Classifier-STARTER_Solution.ipynb  # Main project notebook
├── 🔧 setup_environment.sh                      # Complete conda environment setup
├── 📝 activate_env.sh                          # Environment activation script (auto-generated)
├── 📋 environment.yml                          # Conda environment specification (auto-generated)
├── 💾 cifar10_simple_model.pth                 # Trained simple CNN model
├── 💾 cifar10_complex_model.pth                # Trained residual CNN model
├── 📁 CIFAR_10_data/                           # Dataset directory (auto-created)
│   ├── cifar-10-python.tar.gz                  # Original dataset
│   └── cifar-10-batches-py/                    # Extracted data
├── 📋 .gitignore                               # Git ignore rules
├── 📋 LICENSE                                  # MIT license
└── 📋 README.md                                # This documentation
```

## 🧠 Model Training Process

### 1. Data Preprocessing
- **Normalization**: Mean=(0.5, 0.5, 0.5), Std=(0.5, 0.5, 0.5)
- **Augmentation**: Random horizontal flip, rotation (±10°)
- **Resizing**: 30×30 pixels for consistent input
- **Train/Validation Split**: 80/20 ratio

### 2. Training Configuration
- **Optimizer**: Adam optimizer
- **Loss Function**: Negative Log Likelihood (NLLLoss)
- **Learning Rates**: 0.0005 (Simple), 0.005 (Residual)
- **Early Stopping**: Based on validation accuracy
- **Device Optimization**: Automatic GPU/CPU detection

### 3. Training Features
- **Progress Tracking**: Real-time loss and accuracy monitoring
- **Visualization**: Training curves and validation metrics
- **Model Persistence**: Automatic model saving
- **Performance Analysis**: Comprehensive evaluation metrics

## 📈 Key Enhancements Made

### Original Submission (September 11, 2024)
- ✅ Basic CNN implementation
- ✅ CIFAR-10 data loading and preprocessing
- ✅ Training loop with validation
- ✅ Model evaluation and testing

### Enhanced Version Improvements
1. **🚀 Advanced Architecture**
   - Added Residual CNN with skip connections
   - Implemented batch normalization
   - Enhanced regularization techniques

2. **⚡ Cross-Platform GPU Support**
   - Comprehensive device detection
   - CUDA support for NVIDIA GPUs
   - MPS support for Apple Silicon
   - Intelligent CPU fallback

3. **🔧 Production Environment**
   - Automated conda environment setup
   - Cross-platform compatibility script
   - Dependency management
   - Easy activation workflow

4. **📊 Enhanced Monitoring**
   - Detailed system information logging
   - GPU memory usage tracking
   - Training progress visualization
   - Performance benchmarking

5. **📝 Documentation & Usability**
   - Comprehensive README with one-command setup
   - Simplified project structure
   - Performance benchmarks and comparisons
   - Clear setup and usage instructions

6. **🔧 Streamlined Environment Management**
   - Single script for complete conda environment setup
   - Automatic environment testing and verification
   - Cross-platform compatibility (macOS, Linux, Windows)
   - Auto-generated environment.yml for reproducibility

## 💡 Technical Innovation

### Residual Architecture Implementation
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # Skip connection implementation for better gradient flow
        # Enables training of deeper networks effectively
```

### Smart Device Detection
```python
def get_optimal_device():
    # Automatic detection of CUDA, MPS, or CPU
    # Cross-platform optimization for maximum performance
```

## 🎓 Educational Value

This project demonstrates:
- **Deep Learning Fundamentals**: CNN architecture design
- **PyTorch Proficiency**: Model implementation and training
- **GPU Programming**: Efficient device utilization
- **MLOps Practices**: Environment management and reproducibility
- **Performance Optimization**: Architecture tuning and evaluation

## 🏅 Comparison to Benchmarks

| Method | Accuracy | Year | Complexity |
|--------|----------|------|------------|
| **Our Residual CNN** | **78%+** | **2024** | **Medium** |
| Detectocorp Benchmark | 70% | - | Unknown |
| Deep Belief Networks | 78.9% | 2010 | High |
| Maxout Networks | 90.6% | 2013 | Very High |
| Wide ResNets | 96.0% | 2016 | Extreme |

## � Troubleshooting

### Common Issues

**❌ "conda: command not found"**
```bash
# Install Miniconda first
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**❌ "Permission denied" when running setup script**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

**❌ Environment activation fails**
```bash
# Reload conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cifar10-classifier
```

**❌ GPU not detected**
- **NVIDIA**: Ensure CUDA drivers are installed
- **Apple Silicon**: Update to macOS 12.3+ for MPS support
- **Fallback**: CPU training will work but be slower

## �🔍 Keywords & Tags

`machine-learning` `deep-learning` `computer-vision` `pytorch` `cnn` `image-classification` `cifar10` `udacity` `nanodegree` `residual-networks` `gpu-acceleration` `mps` `cuda` `conda` `automated-setup` `one-command-setup` `jupyter` `python` `neural-networks` `convolutional-neural-networks` `data-science` `artificial-intelligence` `cross-platform`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Udacity** for providing the comprehensive Machine Learning curriculum
- **PyTorch Team** for the excellent deep learning framework
- **CIFAR-10 Dataset** creators for the benchmark dataset
- **Research Community** for foundational CNN and ResNet architectures

## 📞 Contact

For questions or collaboration opportunities, please reach out through the repository's issue tracker.

---

*Built with ❤️ for the Udacity Machine Learning Nanodegree Program*

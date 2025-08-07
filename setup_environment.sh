#!/bin/bash

# CIFAR-10 Image Classifier - Conda Environment Setup Script
# This script creates and configures a complete conda environment for the CIFAR-10 Image Classifier project
# Compatible with macOS, Linux, and Windows

echo "🚀 CIFAR-10 Image Classifier - Environment Setup"
echo "=================================================="

# Define environment name
ENV_NAME="cifar10-classifier"

# Function to test environment
test_environment() {
    echo "🧪 Testing environment setup..."
    
    # Test imports
    python -c "
import sys
import platform
print(f'✅ Python {platform.python_version()}')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    
    # Test device availability
    if torch.cuda.is_available():
        print(f'🚀 CUDA available: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f'🍎 MPS (Apple Silicon) available')
    else:
        print(f'🔄 Using CPU')
    
    import torchvision, matplotlib, numpy, tqdm
    print('✅ All packages imported successfully!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    return 1
    
print('🎉 Environment test passed!')
return 0
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Environment verification successful!"
        return 0
    else
        echo "❌ Environment verification failed!"
        return 1
    fi
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "📥 Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "🔍 Conda found: $(conda --version)"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "�️  Removing existing ${ENV_NAME} environment..."
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment with Python 3.9
echo "📦 Creating conda environment: ${ENV_NAME}"
conda create -n $ENV_NAME python=3.9 -y

# Activate the environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Detect OS and install appropriate PyTorch version
echo "🔥 Installing PyTorch with optimal configuration..."
OS_TYPE=$(uname -s)
case $OS_TYPE in
    "Darwin")
        echo "🍎 macOS detected - Installing PyTorch with MPS support"
        conda install pytorch torchvision torchaudio -c pytorch -y
        ;;
    "Linux")
        echo "🐧 Linux detected - Installing PyTorch with CUDA support"
        # Check if NVIDIA GPU is available
        if command -v nvidia-smi &> /dev/null; then
            echo "� NVIDIA GPU detected - Installing CUDA version"
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        else
            echo "🔄 No NVIDIA GPU detected - Installing CPU version"
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        fi
        ;;
    *)
        echo "🪟 Windows/Other OS - Installing CPU version"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ;;
esac

# Install additional required packages
echo "📚 Installing additional packages..."
conda install -y \
    matplotlib \
    numpy \
    jupyter \
    ipykernel \
    tqdm

# Install the environment as a Jupyter kernel
echo "🔗 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name="CIFAR-10 Classifier"

# Test the environment
echo ""
test_environment

# Create activation script for easy future use
echo ""
echo "📝 Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# CIFAR-10 Classifier Environment Activation Script

ENV_NAME="cifar10-classifier"

# Function to activate conda environment
activate_conda_env() {
    if command -v conda &> /dev/null; then
        source $(conda info --base)/etc/profile.d/conda.sh
        if conda env list | grep -q "^${ENV_NAME} "; then
            conda activate $ENV_NAME
            echo "✅ Environment '${ENV_NAME}' activated!"
            echo ""
            echo "🎯 Available commands:"
            echo "   jupyter notebook CIFAR-10_Image_Classifier-STARTER.ipynb"
            echo "   python test_environment.py"
            echo ""
            echo "🔧 To deactivate: conda deactivate"
        else
            echo "❌ Environment '${ENV_NAME}' not found!"
            echo "💡 Run './setup_environment.sh' to create it."
            return 1
        fi
    else
        echo "❌ Conda not found!"
        return 1
    fi
}

activate_conda_env
EOF

chmod +x activate_env.sh

# Create environment export for reproducibility
echo "� Exporting environment specification..."
conda env export > environment.yml

# Success message
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🎯 Quick Start:"
echo "   source ./activate_env.sh"
echo "   jupyter notebook CIFAR-10_Image_Classifier-STARTER.ipynb"
echo ""
echo "🔧 Manual activation:"
echo "   conda activate $ENV_NAME"
echo ""
echo "📊 Environment details:"
echo "   Name: $ENV_NAME"
echo "   Python: $(python --version 2>&1 | cut -d' ' -f2)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Installation check needed')"
echo ""
echo "🔄 To recreate this environment elsewhere:"
echo "   conda env create -f environment.yml"

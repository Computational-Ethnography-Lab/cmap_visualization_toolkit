#!/bin/bash
# Installation script for CMAP Visualization Toolkit

# Create and activate conda environment
echo "Creating conda environment..."
conda create -y --name cmap_visualization_toolkit python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cmap_visualization_toolkit

# Install packages from requirements.txt
echo "Installing packages..."
conda install -y jupyter
pip install -r requirements.txt

# Download NLTK resources
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); print('NLTK resources downloaded successfully!')"

echo "Installation complete! You can now run: jupyter notebook visulization_toolkit_final.ipynb"

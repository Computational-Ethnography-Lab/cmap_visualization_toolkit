# CMAP Visualization Toolkit

An easy-to-use toolkit for visualizing patterns in qualitative data, helping researchers see and share connections between words, concepts and themes alongside in-depth accounts.

## Table of Contents
- [Overview](#overview)
- [What This Toolkit Does](#what-this-toolkit-does)
- [Installation](#installation)
  - [One-Command Installation (Easiest Method)](#one-command-installation-easiest-method)
  - [Step-by-Step Installation](#step-by-step-installation)
  - [Manual Installation](#manual-installation)
    - [Prerequisites](#prerequisites)
    - [Using Conda CLI](#using-conda-cli)
    - [Using Anaconda Navigator GUI](#using-anaconda-navigator-gui)
- [Using the Toolkit](#using-the-toolkit)
  - [First Time Setup](#first-time-setup)
  - [Regular Usage](#regular-usage)
- [Using Your Own Data](#using-your-own-data)
  - [Data Structure](#data-structure)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)
- [License](#license)
- [Training Resources](#training-resources)
- [References](#references)
- [Disclosure](#disclosure)

## Overview

The CMAP Visualization Toolkit helps you examine patterns in text data (e.g. qualitative interviews, fieldnotes, and documents) using visual tools. It provides a bridge to combine computational text analysis with qualitative research methods to identify patterns and visualize relationships (based on academic work by Abramson et al. 2018, 2025).

The notebook included with this toolkit explains all the methods in detail.

## What This Toolkit Does

This toolkit gives you several ways to analyze and visualize your text data:

1. **Basic Descriptive Statistics**: Get counts, frequencies, and key metrics about your data

2. **Data Validation Tools**: Check that your data is formatted correctly and identify potential issues

3. **Word-Level Visualizations**:
   - **Word Clouds**: See which words appear most often in your text
   - **t-SNE Maps**: View how words relate to each other in 2D space
   - **Word Relationship Heatmaps**: See which words appear together frequently
   - **Semantic Networks**: Visualize connections between related words

4. **Code-Level Visualizations**:
   - **Code Co-occurrence Heatmaps**: See which qualitative codes appear together
   - **Code Networks**: Visualize relationships between different themes or codes

All analysis happens on your computer - no data is sent anywhere else.

You can use this software with commercial qualitative data analysis by using the export process described here [blog placeholder]

## Installation

**⚠️ IMPORTANT**: This toolkit requires Anaconda or Miniconda to be installed on your system. If you don't have it yet, [download and install Anaconda](https://www.anaconda.com/products/distribution) or [download and install Miniconda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

### One-Command Installation (Easiest Method)

For the simplest installation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit.git
   cd cmap_visualization_toolkit
   ```
   
When prompted by GitHub:
	•	Username: Enter your GitHub username (e.g., your-username).
	•	Password: ⚠This is **not your GitHub login password**. GitHub now requires a ** Personal Access Token (PAT)** instead.

**How to get a Personal Access Token (PAT):**
	1.	Log into GitHub.
	2.	Go to Settings -> Developer settings -> Personal access tokens -> Tokens (classic).
	3.	Click Generate new token (classic), give it a name, set an expiration, and check the box for repo.
	4.	Copy the generated token (you will only see it once).
	5.	When Git asks for your password, paste this token.

Tip: To avoid typing your PAT every time, you can save it using Git Credential Manager or macOS Keychain.

2. **Run the installation script**:

   **For macOS/Linux**:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

   **For Windows** (in Anaconda Prompt):
   ```bash
   conda create -y --name cmap_visualization_toolkit python=3.11
   conda activate cmap_visualization_toolkit
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); print('NLTK resources downloaded successfully!')"
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook visulization_toolkit_final.ipynb
   ```

### Step-by-Step Installation

If the one-command method doesn't work, try these step-by-step commands:

```bash
# 1. Clone the repository
git clone https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit.git
cd cmap_visualization_toolkit

# 2. Create and activate conda environment
conda create -y --name cmap_visualization_toolkit python=3.11
conda activate cmap_visualization_toolkit

# 3. Install Jupyter (to ensure we have it before other packages)
conda install -y jupyter

# 4. Install packages from requirements.txt
pip install -r requirements.txt

# 5. Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 6. Launch Jupyter Notebook
jupyter notebook visulization_toolkit_final.ipynb
```

This installation method ensures all packages are installed with the correct versions specified in the `requirements.txt` file.

### Manual Installation

If you prefer to install step by step or need more control over the process:

#### Prerequisites

Before starting, you need to install Anaconda, which is free software that helps manage Python packages.

1. **Download Anaconda**:
   - Go to the [Anaconda website](https://www.anaconda.com/products/distribution)
   - Click the "Download" button
   - Choose the version for your computer (Windows, Mac, or Linux)

2. **Install Anaconda**:
   - Double-click the downloaded file
   - Follow the on-screen instructions
   - Accept the default options if you're unsure

#### Using Conda CLI

For users comfortable with command line:

1. **Open Terminal or Command Prompt**:
   - Windows: Open "Anaconda Prompt" from Start menu
   - Mac/Linux: Open Terminal app

2. **Create and Set Up Environment**:
   ```bash
   # Create a new environment
   conda create --name cmap_visualization_toolkit python=3.11
   
   # Activate the environment
   conda activate cmap_visualization_toolkit
   
   # Get the code
   git clone https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit.git
   cd cmap_visualization_toolkit
   
   # Install Jupyter
   conda install -y jupyter
   
   # Install other packages with version constraints
   pip install -r requirements.txt
   
   # Download NLTK resources (standard language processing datasets)
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); print('NLTK resources downloaded successfully!')"
   ```

#### Using Anaconda Navigator GUI

For users who prefer a visual interface:

1. **Open Anaconda Navigator**:
   - Windows: Click Start menu → Anaconda Navigator
   - Mac: Open Applications folder → Anaconda Navigator
   - Linux: Open terminal and type `anaconda-navigator`

2. **Create a New Environment**:
   - Click on "Environments" tab on the left side
   - Click "Create" button at the bottom
   - Type `cmap_visualization_toolkit` as the name
   - Select Python 3.11 from the dropdown
   - Click "Create" button

3. **Install Jupyter**:
   - With your new environment selected, go to the "Home" tab
   - Select your new environment from the dropdown menu
   - Install Jupyter Notebook by clicking "Install"

4. **Open Terminal in Your Environment**:
   - Go back to "Environments" tab
   - Click on your `cmap_visualization_toolkit` environment
   - Click the play button (▶) and select "Open Terminal"
   - In the terminal, run:
   
   ```bash
   # Get the code
   git clone https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit.git
   cd cmap_visualization_toolkit
   
   # Install packages
   pip install -r requirements.txt
   
   # Download NLTK resources (standard language processing datasets)
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); print('NLTK resources downloaded successfully!')"
   ```

## Using the Toolkit

### First Time Setup

1. **Start Your Environment** (if not already activated):
   ```bash
   conda activate cmap_visualization_toolkit
   ```

2. **Open the Notebook**:
   ```bash
   jupyter notebook visulization_toolkit_final.ipynb
   ```
   
   If using VS Code:
   1. Open VS Code
   2. Click "File" → "Open Folder" and select the cmap_visualization_toolkit folder
   3. Find and double-click on `visulization_toolkit_final.ipynb`
   4. When prompted, select the `cmap_visualization_toolkit` kernel

3. **Run the Code**:
   1. Click on the first gray box (called a "cell")
   2. Click the "Run" button (triangle symbol ▶) or press Shift+Enter
   3. Wait for it to finish (when the * symbol disappears)
   4. Move to the next cell and repeat

### Regular Usage

1. **Start Your Environment**:
   - Using Navigator: Open Anaconda Navigator, click your environment, then click "▶" and "Open Terminal"
   - Using command line: Open terminal and type `conda activate cmap_visualization_toolkit`

2. **Open the Notebook**:
   ```bash
   jupyter notebook visulization_toolkit_final.ipynb
   ```

3. **Run Each Section**:
   1. Click on a cell
   2. Press the Run button (▶) or Shift+Enter
   3. Continue through all cells in order

4. **Save Results**:
   - To save an image: Right-click on it and select "Save Image As..."
   - To copy text: Highlight it and press Ctrl+C (Windows) or Cmd+C (Mac)

## Using Your Own Data

To analyze your own text data:

1. **Prepare Your Data**:
   - Create a CSV file with at least a column called `text`
   - Optionally add a column called `project` to group texts
   - Save it in the `data` folder

2. **Change the File Path**:
   - In the notebook, find the cell that loads data
   - Change the file name to your CSV file name
   - Run the cells in order

3. **Adjust Settings**:
   - Word clouds: Change keywords to find specific topics
   - Networks: Adjust threshold values to show more/fewer connections
   - Heatmaps: Change clustering method (1=RoBERTa, 2=Jaccard, 3=PMI, 4=TF-IDF)

### Data Structure

For technically minded users, here's the complete schema for data files:

```python
# Updated schema with Python typing
schema = {
    "project": str,         # List project 
    "number": str,          # Position information
    "reference": int,       # Position information
    "text": str,            # Content, critical field: must not be empty
    "document": str,        # Data source, Critical field: must not be empty
    "old_codes": list[str], # Optional: codings, must be a list of strings
    "start_position": int,  # Position information
    "end_position": int,    # Position information
    "data_group": list[str],# Optional, to differentiate document sets: Must be a list of strings
    "text_length": int,     # Optional: NLP info
    "word_count": int,      # Optional: NLP info
    "doc_id": str,          # Optional: NLP info, unique paragrah level identifier
    "codes": list[str]      # Critical for analyses with codes, Must be a list of strings
}
```

**Critical Fields**:
- `text`: Main content field - cannot be empty
- `document`: Source information - cannot be empty
- `codes`: Required for code-based analyses - must be a list of strings

**Important Notes**:
- Lists (like `codes` and `data_group`) must be proper Python lists, not strings that look like lists
- If you're exporting from qualitative data analysis software, ensure you convert any code fields to proper lists
- The toolkit will validate your data structure and provide error messages for common issues

## Troubleshooting

Here are solutions to common issues you might encounter:

1. **Installation Script Issues**:
   - If the installation script doesn't work, try the step-by-step commands in the [Step-by-Step Installation](#step-by-step-installation) section
   - For permission issues with `install.sh`, run: `chmod +x install.sh` before executing

2. **Package Version Conflicts**:
   - If you see version compatibility errors, try installing without version specifications: `pip install -r requirements.txt --no-deps`
   - For Mac with Apple Silicon (M1/M2/M3), you may need: `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`

3. **CUDA/GPU Issues with PyTorch**:
   - If you encounter CUDA errors, you might need a specific torch version: `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
   - For CPU-only: `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu`

4. **Memory Errors**:
   - If you get "out of memory" errors, try processing smaller batches of data
   - Close other applications to free up system memory

5. **Import Errors**:
   - Make sure your directory structure is correct with the `function` folder at the same level as the notebook
   - Check that all packages are installed correctly

6. **Visualization Issues**:
   - If plots are not displaying correctly, try running `%matplotlib inline` in a notebook cell
   - For interactive plots, run `pip install ipywidgets` and then `jupyter nbextension enable --py widgetsnbextension`

7. **Data Format Issues**:
   - If you see errors related to data types, ensure your CSV has the correct format per the schema
   - Common issue: Make sure `codes` and `data_group` are proper lists, not strings
   - Fix: Use `df['codes'] = df['codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)` to convert string representations to lists

For more detailed troubleshooting, please visit the [project website](https://computationalethnography.org/) or open an issue on the GitHub repository.

## Uninstallation

To remove the CMAP Visualization Toolkit from your system:

1. **Remove the Environment**:
   ```bash
   # Deactivate the environment if it's currently active
   conda deactivate
   
   # Remove the environment and all its packages
   conda env remove --name cmap_visualization_toolkit
   ```

2. **Delete the Code**:
   ```bash
   # Navigate up one directory (if you're in the project directory)
   cd ..
   
   # Remove the project directory
   rm -rf cmap_visualization_toolkit
   ```

3. **Clean Conda Cache** (Optional):
   ```bash
   # Remove unused packages and caches
   conda clean --all
   ```

This will completely remove all toolkit components from your system.

## License

BSD 3-Clause License

Copyright (c) 2025 Computational Ethnography Lab (Abramson et al.)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the Computational Ethnography Lab nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**Important**: If you use this software, please cite as: "Abramson, Corey, Yuhan (Victoria) Nian, and Zhuofan Li. 2025. CMAP Visualization Toolkit. https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit."

Key contributors: Yuhan (Victoria) Nian, Zhuofan Li, and others.

No warranty is provided. If you want to contribute, please email corey.abramson@rice.edu.

## Training Resources

### Anaconda Setup Videos

- **Anaconda Navigator (GUI) Setup**: [Getting Started with Anaconda Navigator](https://www.youtube.com/watch?v=5mDYijMfSzs)
- **Conda Command Line (CLI) Setup**: [Getting Started with Conda](https://www.youtube.com/watch?v=23aQdrS58e0)

For more detailed information, refer to the [Anaconda Documentation](https://docs.anaconda.com/).

## References

This toolkit builds on academic work combining computational text analysis with qualitative research methods (Abramson et al. 2018, 2025). Please see the [official project website](https://computationalethnography.org/) for additional resources and related research papers.

## Disclosure

LLms were used to check for errors (primarily claude-sonnet), and help annotate code and documentation

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
- [Training Resources](#training-resources)
- [References](#references)
- [Policies](#policies)
  - [License](#license)
  - [Authors](#authors)
  - [Acknowledgments](#acknowledgments)
  - [Data Ethics](#data-ethics)
  - [Disclaimer](#disclaimer)

## Overview
The CMAP Visualization Toolkit offers a free suite of open-source tools to analyze and visualize text data: including fieldnotes, in-depth interview transcripts, historical documents, web-pages, and other forms of non-numeric information. It is designed for scholars working with qualitative methods, who have an interest in the possibilities for pattern analysis, data visualization, and identifying alternative explanations found in computational social science's (CSS).

The CMAP (Cultural Mapping and Pattern Analysis) tool is free, open-source and produced by the [Computational Ethnography Lab](https://github.com/Computational-Ethnography-Lab) at Rice University.

---

### Notebook Version & Local Installation

For an introduction, in an easy to use online version (not for sensitive data) check out the collab version [here](https://colab.research.google.com/drive/1n90EDMSiXhIaOULUMPJ4W4hqdZCh1NQw?usp=sharing)
---

## What This Toolkit Does

This notebook introduces elements of visualizing text data from qualitative sources and provides tools for:
-   Validating text data
-   Generating basic text statistics
-   Charting concepts
-   Visualizing themes
-   Drawing comparisons at the level of words, codes, concepts, and documents
-   Allowing analyses across subsets of data (e.g. examining variation by neighborhood, occupation, time)

CMAP Visualization Toolkit supports advanced analytic methods that are appropriate for computational text analysis and can be used alongside in-depth readings-- including co-occurence, clustering and embedding apporaches-- with visuals such as heatmaps, t-SNE dimensional reducation plots (like a scatter plot, with words), semantic networks, word clouds, and more. The examples are designed to work with common qualitative data sources and allow granular analysis that mirror qualitative practices (at the level of words, sentences, paragraphs), yet are scalable for large datasets produced by teams.

#### Sample Visualizations
*Examples from this toolkit using public data on scientists' careers.*

<div class="grid">
  <div class="grid-item">
    <img src="https://github.com/Computational-Ethnography-Lab/.github/raw/95529a5a1ffa938274ac5b4b912dbf99e26fd572/profile/images/heatmap_visualization.png" class="plot-img">
    <div class="caption"><em>A heatmap visualizing patterned variation in interviews.</em></div>
  </div>

  <div class="grid-item">
    <img src="https://github.com/Computational-Ethnography-Lab/.github/raw/95529a5a1ffa938274ac5b4b912dbf99e26fd572/profile/images/semantic_network.png" class="plot-img">
    <div class="caption"><em>A semantic network of concepts in field notes and interviews.</em></div>
  </div>

  <div class="grid-item">
    <img src="https://github.com/Computational-Ethnography-Lab/.github/raw/95529a5a1ffa938274ac5b4b912dbf99e26fd572/profile/images/tsne.png" class="plot-img">
    <div class="caption"><em>A t-SNE plot for document clustering.</em></div>
  </div>

  <div class="grid-item">
    <img src="https://github.com/Computational-Ethnography-Lab/.github/raw/95529a5a1ffa938274ac5b4b912dbf99e26fd572/profile/images/wordcloud.png" class="plot-img">
    <div class="caption"><em>A word cloud with user-defined concept groups.</em></div>
  </div>
</div>

** Read more about the approach [here](https://github.com/Computational-Ethnography-Lab)**

## Installation

**⚠️ IMPORTANT**: This toolkit requires Anaconda or Miniconda to be installed on your system. If you don't have it yet, [download and install Anaconda](https://www.anaconda.com/products/distribution) or [download and install Miniconda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

### One-Command Installation (Easiest Method) 

For the simplest installation, follow these steps:

**If you are using anaconda:**
  -  Open the anaconda program
  -  Left-click 'Environments' (left side of the interface)
  -  Left-click (left click the arrow next to base(root)-> cick open terminal.
  -  In the terminal window that opens, cut and paste the text in steps 1-3.
    -   Make sure to use the version for your system (windows, mac, or linux)
      -   _Note: some windows versions require pasting by right-clicking instead of ctrl-v_


1. **Clone the repository**:
   ```bash
   git clone https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit.git
   cd cmap_visualization_toolkit
   ```
   

2. **Run the installation script**:
- This sets up the python packages needed to run the toolkit.
  
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
   jupyter notebook visualization_toolkit_final.ipynb
   ```
**Now, you should be in the toolkit! You can run each block by clicking the run buton (triangular arrow) for each cell. The program explains what each cell does, and what can be edited for analysis/**

**If you get 'unable to compare versions error' check the kernel, in the top right. It should say something like visualization toolkit."
** You can also open the .ipynb file in vscode or another development environment, just check the kernel**

### Step-by-Step Installation

If the one-command method doesn't work, or you want more granular control at the command line, try these step-by-step commands:

First, open your terminal. Then paste the following into the command line interface in order.

```bash
# 1. Create and activate conda environment (assuming a conda version)
conda create -y --name cmap_visualization_toolkit python=3.11
conda activate cmap_visualization_toolkit
conda install git

# 2. Clone the repository
git clone https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit.git
cd cmap_visualization_toolkit

# 3. Install Jupyter (to ensure we have it before other packages)
conda install -y jupyter

# 4. Install packages from requirements.txt
pip install -r requirements.txt

# 5. Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 6. Launch Jupyter Notebook
jupyter notebook visulization_toolkit_final.ipynb

# 6b. If you would rather use VS code or another integrated development enviornment, simply open the .ipynb file in that application.
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
   - IF using Navigator: Open Anaconda Navigator, click your environment, then click "▶" and "Open Terminal"
   - IF using command line: Open terminal and type `conda activate cmap_visualization_toolkit`, and switch to the directory with the toolkid 'cd cmap_visualization_toolkit'

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

For technically minded users, here's the complete schema for data files (See Abramson et al. 2025):

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
```python
  example (simulated, not actual data):
  {
    "project": "engineer_interviews",  
    "number": "675:113" #For reconstructing with QDA software  
    "reference": 244 #For reconstructing with QDA software  
    "text": "EG002: So the thing is..." # Actual paragraph level text,
    "document": "INTV_EG002_20250801.txt" # name of document, in which text is found
    "start_position": 244 #For reconstructing with QDA software  
    "end_position": 248 #For reconstructing with QDA software  
    "data_group": ["data_type_cg_interviews","interview"] # For subsetting by data type, or characteristic
    "text_length": 441 # nlp
    "word_count": 82 # nlp
    "doc_id": "EG002_76426", #unique ID for text segment in file, to reconstruct or link to raw data, sequential
    "codes": ["narrative_life", "education_perceptions"] # qualitative codes, for concepts, themes, variables
    "old_codes": [
      "science_belief",
      "subject_speech_all"
    ] # qualitative codes, for concepts, themes, variables; archived to silo and reduce clutter

  },//
```

**Critical Fields**:
- `text`: Main content field - cannot be empty
- `document`: Source information - cannot be empty
- `codes`: Required for code-based analyses - must be a list of strings if used

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

## Training Resources

### Anaconda Setup Videos

- **Anaconda Navigator (GUI) Setup**: [Getting Started with Anaconda Navigator](https://www.youtube.com/watch?v=5mDYijMfSzs)
- **Conda Command Line (CLI) Setup**: [Getting Started with Conda](https://www.youtube.com/watch?v=23aQdrS58e0)

For more detailed information, refer to the [Anaconda Documentation](https://docs.anaconda.com/).

## References

This toolkit builds on academic work combining computational text analysis with qualitative research methods (Abramson et al. 2018, 2025). Please see the [lab repo](https://github.com/Computational-Ethnography-Lab) for additional resources and related research papers.

# Policies
## License
See [LICENSE.md](./LICENSE.md).
BSD 3-Clause License
Copyright (c) 2025 Computational Ethnography Lab (Abramson et al.)

**Important**: If you use this software, please cite as: "Abramson, Corey and Yuhan (Victoria) Nian. 2025. CMAP Visualization Toolkit. https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit."

Key contributors: Yuhan (Victoria) Nian, Zhuofan Li, Tara Prendergast, Jakira Silas, and others.

No warranty is provided. If you want to contribute, please email corey.abramson@rice.edu.

### Authors
- Corey M. Abramson, Ph.D. — Associate Professor of Sociology, Rice University; PI, Computational Ethnography Lab; Co-Director, CIISR  
- Yuhan (Victoria) Nian — Undergraduate Student, Statistics & Data Science, Rice University  
- Zhuofan Li, Ph.D. — Assistant Professor of Sociology, Virginia Tech  
- Tara Prendergast — Ph.D. Candidate, School of Sociology, University of Arizona  
- Jakira Silas — Graduate Student, Sociology, Rice University

  
### Acknowledgments
We thank all contributors to the Computational Ethnography Lab GitHub repository for their valuable feedback, testing, and improvements. This includes everyone with contributions to the public repository, in addition to the key authors listed above. Special thanks to the broader Computational Ethnography Lab community for ongoing support.

### Data Ethics
See [data_ethics.md](./DATA_ETHICS.md) for details on ethical considerations, including data anonymization, consent, and use restrictions.

### Disclaimer
See [DISCLAIMER.md](./DISCLAIMER.md) for important legal and usage disclaimers. 
### Disclosures
- LLMs (primarily Claude-Sonnet) were used to check for errors and help annotate code.  
- If you find errors or are interested in collaboration, contact corey.abramson@rice.edu.  
- This free software carries no warranty or guarantee—like life.

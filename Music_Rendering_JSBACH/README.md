# Expressive Performance Generation with VirtuosoNet

**Project:** Expressive Music Performance (Schumann Variation V)
**Group/Student:** JS Bach
**Date:** January 2026


## 1. Project Overview
This submission contains the code, report, and generated results for the "Expressive Rendering" project. The objective was to generate a human-like, expressive piano performance of **Robert Schumann’s Variation V** (from *Symphonic Etudes*) using the **VirtuosoNet** Hierarchical Attention Network (HAN).

**Key Achievement:**
Successfully generated a valid **Adagio** interpretation (63 seconds) from a score traditionally performed as a *Presto* Etude (39 seconds) by leveraging Composer Embedding Style Transfer (Schumann $\to$ Chopin).


## 2. Submission Contents

### **Core Files**
* `Report.ipynb`: The main project report (Jupyter Notebook) containing the methodology, analysis of results, and phase-space visualization.
* `Yasaman_Shokri_Schumann.mid`: The final generated expressive performance (MIDI).
* `analysis_plot.png`: Phase-space trajectory plot (Tempo vs. Dynamics) used in the report.

### **Code & Environment**
* `model_run.py`: The primary inference script.
* `virtuosoNet/`: Source code directory containing the model architecture and auxiliary classes.
* `environment.yml`: Conda environment definition file for reproducing the setup.


## 3. Setup Instructions

### **3.1 Environment Setup**
To reproduce the Python environment, use the provided YAML file:
```bash
conda env create -f environment.yml
conda activate expressive_project
```

### **3.2 Model Weights (Critical)**
Due to file size restrictions (>100MB), the pre-trained model weights are not included in this zip file. You must place the weight files in the root directory of this project for the code to run.

Required File: han_ar_note_best.pth.tar (or prime_isgn_best.pth.tar as a fallback).

Download Source: These can be downloaded from the official VirtuosoNet repository or the course resources.


### **4. Reproduction (Generating the Submission File)**
To reproduce the specific "Romantic Adagio" interpretation submitted in this project, run the command below.

Methodology: We utilize the Chopin composer embedding (to maximize rubato and lyricism) and enforce a 60 BPM tempo constraint to meet the duration requirement.


```bash 
python3 model_run.py \
  -mode=test \
  -code=han_ar_note \
  -path=./test_pieces/schumann_variation/ \
  -comp=Chopin \
  -tempo=60
```

* Output Location: The generated MIDI and PNG files will appear in the test_result/ folder.

* Note on Weights: Ensure the --resume argument in model_run.py defaults to the weight file you have downloaded (e.g., han_ar_note_best.pth.tar).


### **5. References**
VirtuosoNet: Jeong, D., et al. (2019). VirtuosoNet: A Hierarchical Attention-based Model for Generating Expressive Piano Performance. ISMIR 2019.

Dataset: Yamaha e-Piano Competition Dataset.
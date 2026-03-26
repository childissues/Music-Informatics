# Composer Identification Challenge - Team J. S. Bach

This repository contains the submission for the Composer Identification Challenge for the Musical Informatics course (WS2025).

## Team Members
- Bivek Kumar Sah
- Arian Moradi
- Yasaman Shokriazar
- Calvin Oluyemi

## Contents
- `J. S. Bach_ComposerIdentification.py`: Main submission script.
- `J. S. Bach_Report.ipynb`: Descriptive report in Jupyter Notebook format.
- `bach_model.joblib`: Pre-trained Random Forest model weights.
- `environment.yml`: Conda environment definition.
- `results_composer_identification.csv`: Final predictions for the test set.

## Setup Instructions

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate miws25_bach
   ```

2. **Run the classification script:**
   To generate results for the test set, you can use the provided helper script:
   ```bash
   ./run_challenge.sh
   ```
   
   Alternatively, run the python command directly:
   ```bash
   python "J. S. Bach_ComposerIdentification.py" -i "Dataset-20260125/scores_composer_identification" -o "results_composer_identification.csv"
   ```
   Replace the paths if your data location differs.

3. **View the report:**
   Open `J. S. Bach_Report.ipynb` in Jupyter Notebook or JupyterLab to read the project description and results discussion.

## Methodology
Our approach uses a Random Forest classifier trained on high-level musical features extracted from MusicXML files via `partitura`. Features include:
- Pitch and duration distributions.
- Melodic interval distributions.
- Global statistics like average pitch, duration, and polyphonic density.

We achieved a 5-fold cross-validation accuracy of approximately 68.7% on the training set.

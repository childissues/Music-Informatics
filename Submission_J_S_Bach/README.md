# Piece Difficulty Estimation - Team J. S. Bach

## Team Members
- Bivek Kumar Sah
- Arian Moradi
- Yasaman Shokriazar
- Calvin Oluyemi

## Project Structure
- `J_S_Bach_PieceDifficulty.py`: The main script for the challenge submission.
- `Report.ipynb`: The project report (Blog Post).
- `features.py`: Helper module for feature extraction using `music21`.
- `model.joblib`: The trained Random Forest model.
- `feature_columns.json`: List of feature columns used by the model.
- `environment.yaml`: Conda environment definition.

## Setup Instructions
1. Install the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate miws25
   ```

2. To run the challenge script:
   ```bash
   python J_S_Bach_PieceDifficulty.py -i <path_to_scores> -o <output_csv_path>
   ```
   Example:
   ```bash
   python J_S_Bach_PieceDifficulty.py -i Dataset-20260125/scores_difficulty_estimation -o results.csv
   ```

3. To view the report:
   Open `Report.ipynb` in Jupyter Notebook or JupyterLab.

## Model Training
The model was trained using `train.py` (included for reference if needed, but not required to run the submission). It uses feature extraction based on pitch range, note density, and polyphony.

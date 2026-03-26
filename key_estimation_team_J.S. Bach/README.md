# Key Estimation Challenge - Team J. S. Bach

## Team Members
- Bivek Kumar Sah
- Arian Moradi
- Yasaman Shokriazar
- Calvin Oluyemi

## Setup

1.  **Clone the repository** (if you haven't already).
2.  **Create and activate the environment**:
    ```bash
    conda env create -f environment.yaml
    conda activate miws25
    ```

## Usage

### Run Key Estimation Challenge Script

To run the key estimation script on the dataset:

```bash
python J_S_Bach_KeyEstimation.py -i <path_to_data_directory> -o results_key_estimation.csv
```

Example:
```bash
python J_S_Bach_KeyEstimation.py -i key_estimation_dataset -o results_key_estimation.csv
```

This will:
1.  Load the dataset from the specified directory.
2.  Evaluate the model on the **training set** (prints Accuracy, F1-score, etc.).
3.  Generate predictions for the **test set** and save them to `results_key_estimation.csv`.

### Jupyter Notebook Report
Open `Report.ipynb` to view the project report, which includes:
- Introduction
- Methods description
- Evaluation results
- Discussion and Critical Reflection

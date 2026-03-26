# Music Alignment Project - Team J. S. Bach

**Members:**
*   Bivek Kumar Sah
*   Arian Moradi
*   Yasaman Shokriazar
*   Calvin Oluyemi

## Setup Instructions

1.  **Create Environment**:
    ```bash
    conda env create -f environment.yaml
    conda activate miws25_jsbach
    ```

2.  **Run Alignment**:
    To run the alignment script on the dataset:
    ```bash
    python JSBach_Alignment.py -c -i path_to_data_directory -o output_directory
    ```
    Example:
    ```bash
    python JSBach_Alignment.py -c -i trainingset -o outputs
    ```

3.  **View Report**:
    Open `JSBach_Report.ipynb` in Jupyter Notebook or Jupyter Lab.
    ```bash
    jupyter notebook JSBach_Report.ipynb
    ```

## File Description
*   `JSBach_Alignment.py`: Main script for alignment. Implements DTW with Numba optimization.
*   `JSBach_Report.ipynb`: Project report and evaluation.
*   `environment.yaml`: Conda environment definition.
*   `verify_results.py`: Auxiliary script used to calculate evaluation metrics.
*   `create_notebook.py`: Script used to generate the report notebook programmatically.

## Algorithm Details
We use Dynamic Time Warping (DTW) on pitch features. To handle arpeggiated chords in the performance, we quantize the performance onset times (window=0.2s) and sort by pitch within those windows. This ensures a more monotonic alignment with the score.

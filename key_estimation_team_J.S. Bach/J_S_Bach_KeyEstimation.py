#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Submission for the key estimation challenge 
for Musical Informatics WS25

Team: J. S. Bach
Members: Bivek Kumar Sah, Arian Moradi, Yasaman Shokriazar, Calvin Oluyemi
"""
import glob
import warnings
import os
import numpy as np
import pandas as pd
import partitura as pt

from typing import Callable, Tuple, Union, List
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Filter partitura warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="partitura.*",
)

# Import from local auxiliary files
from key_profiles import (
    build_key_profile_matrix,
    KEYS,
)
from partitura.utils.music import get_time_units_from_note_array
from key_estimation_challenge import compare_keys


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient

    Parameters
    ----------
    x : np.ndarray
        Input array x
    y : np.ndarray
        Input array y

    Returns
    -------
    c: float
        Pearson's correlation coefficient
    """

    c = np.corrcoef(x, y)[0, 1]
    return c


def estimate_key(
    midi_fn: str,
    pitch_profiles: str = "kk",
    similarity_func: Callable = corr,
) -> str:
    """Estimate Key using the Krumhansl-Schmuckler algorithm

    Parameters
    ----------
    midi_fn : str
        Path to the MIDI file
    pitch_profiles: str
        Pitch profiles to build the key profile matrix.
    similarity_func: Callable
        Function to determine the similarity with the pitch profiles.
        Default is Pearson's correlation.

    Returns
    -------
    key: str
        Key signature of the MIDI file.
    """
    key_profile_matrix = build_key_profile_matrix(pitch_profiles)

    perf = pt.load_performance_midi(midi_fn)
    note_array = perf.note_array()

    _, duration_unit = get_time_units_from_note_array(note_array)

    normalize_distribution = False

    # Get pitch classes
    pitch_classes = np.mod(note_array["pitch"], 12)

    # Compute weighted key distribution
    pitch_distribution = np.array(
        [
            note_array[duration_unit][np.where(pitch_classes == pc)[0]].sum()
            for pc in range(12)
        ]
    )

    if normalize_distribution:
        # normalizing is unnecessary for computing the correlation, but might
        # be necessary for other similarity metrics
        pitch_distribution = pitch_distribution / pitch_distribution.sum()

    # Compute correlation with key profiles
    similarity = np.array(
        [similarity_func(pitch_distribution, kp) for kp in key_profile_matrix]
    )

    key = KEYS[similarity.argmax()]

    return key


def load_key_estimation_dataset(datadir: str) -> np.ndarray:

    train_fns = glob.glob(os.path.join(datadir, "train", "*.mid"))
    test_fns = glob.glob(os.path.join(datadir, "test", "*.mid"))

    labels_train_fn = os.path.join(
        datadir,
        "train",
        "key_estimation_train_gt.csv",
    )

    labels_test_fn = os.path.join(datadir, "test", "key_estimation_test_gt.csv")

    if not os.path.exists(labels_train_fn):
        labels_train = {}
    else:
        labels_train = (
            pd.read_csv(
                labels_train_fn,
                delimiter=",",
            )
            .set_index("filename")["key"]
            .to_dict()
        )

    if not os.path.exists(labels_test_fn):
        labels_test = {}
    else:
        labels_test = (
            pd.read_csv(
                labels_test_fn,
                delimiter=",",
            )
            .set_index("filename")["key"]
            .to_dict()
        )

    return train_fns, test_fns, labels_train, labels_test


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Key estimation - Team J. S. Bach")

    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default="key_estimation_dataset",
        type=str,
        required=False
    )

    parser.add_argument(
        "--outfn",
        "-o",
        help="Output file with results",
        type=str,
        default="results_key_estimation.csv",
    )

    args = parser.parse_args()

    if args.datadir is None:
        if os.path.exists("key_estimation_dataset"):
            args.datadir = "key_estimation_dataset"
            print("No data directory provided. Using default: key_estimation_dataset")
        else:
            parser.print_help()
            raise ValueError("No data directory given and default 'key_estimation_dataset' not found.")

    print(f"Loading data from {args.datadir}...")
    train_fns, test_fns, train_labels, test_labels = load_key_estimation_dataset(
        args.datadir
    )
    print(f"Found {len(train_fns)} training files and {len(test_fns)} test files.")

    ## Test method on training set
    print("\nEvaluating on Training Set...")
    predictions_train = []
    ground_truth_train = []
    for i, midi_fn in enumerate(train_fns):
        pred_key = estimate_key(midi_fn)
        basename = os.path.basename(midi_fn)
        predictions_train.append((basename, pred_key))
        # print(f"{i+1}/{len(train_fns)} {basename}: {pred_key}")
        if basename in train_labels:
            gt_key = train_labels[basename]
            tonal_dist = compare_keys(
                prediction_key=pred_key,
                ground_truth_key=gt_key,
            )
            ground_truth_train.append((pred_key, gt_key, tonal_dist))

    if len(ground_truth_train) > 0:
        ground_truth_train = np.array(
            ground_truth_train,
            dtype=[
                ("preds", "U10"),
                ("gt", "U10"),
                ("tonal_distance", float),
            ],
        )

        accuracy = accuracy_score(
            y_true=ground_truth_train["gt"],
            y_pred=ground_truth_train["preds"],
        )
        f_score = f1_score(
            y_true=ground_truth_train["gt"],
            y_pred=ground_truth_train["preds"],
            average="macro",
        )
        mean_tonal_distance = ground_truth_train["tonal_distance"].mean()

        print("#" * 35)
        print("Key Estimation Results on Training Set\n")
        print(f"   Average Tonal Distance: {mean_tonal_distance:.2f}")
        print(f"   F1-score (macro): {100*f_score:.2f}")
        print(f"   Accuracy: {100*accuracy:.2f}")
        print("#" * 35)

    
    print("\nGenerating Predictions for Test Set...")
    predictions_test = []
    ground_truth_test = []
    for i, midi_fn in enumerate(test_fns):
        pred_key = estimate_key(midi_fn)
        basename = os.path.basename(midi_fn)
        predictions_test.append((basename, pred_key))
        # print(f"{i+1}/{len(test_fns)} {basename}: {pred_key}")
        
    # Save results for challenge submission
    print(f"Saving results to {args.outfn}...")
    with open(args.outfn, "w") as f:
        # Assuming the server expects no header or specific header. 
        # The prompt says "generate a csv file with the results" and Baseline has header 'filename,key'
        f.write("filename,key\n")
        for basename, pred_key in predictions_test:
            f.write(f"{basename},{pred_key}\n")
            
    print("Done.")

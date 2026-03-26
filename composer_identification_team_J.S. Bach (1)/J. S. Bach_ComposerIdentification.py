import os
import argparse
import pandas as pd
import numpy as np
import partitura as pt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from typing import Union, List, Tuple
import joblib

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="partitura.*")

COMPOSER_CLASSES = {
    "Claude Debussy": 0,
    "Franz Liszt": 1,
    "Franz Schubert": 2,
    "Johann Sebastian Bach": 3,
    "Ludwig van Beethoven": 4,
    "Maurice Ravel": 5,
    "Robert Schumann": 6,
    "Sergei Rachmaninoff": 7,
    "Wolfgang Amadeus Mozart": 8,
}

CLASSES_COMPOSER = {v: k for k, v in COMPOSER_CLASSES.items()}

def extract_features(fn: str) -> np.ndarray:
    """
    Extract musical features from a MusicXML file.
    """
    try:
        score = pt.load_musicxml(fn)
        if isinstance(score, pt.score.Score):
            # Focus on the first part for simplicity, or merge if necessary
            part = score.parts[0]
        else:
            part = score

        note_array = part.note_array()
        
        # 1. Pitch Distribution (normalized)
        pitches = note_array['pitch']
        pitch_counts = np.bincount(pitches, minlength=128)
        pitch_dist = pitch_counts / len(pitches)

        # 2. Duration Distribution (normalized)
        durations = note_array['duration_beat']
        # Discretize durations into buckets
        dur_bins = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
        dur_dist = np.histogram(durations, bins=np.append(dur_bins, np.inf))[0] / len(durations)

        # 3. Interval Distribution
        # Sort notes by onset to get intervals
        sorted_notes = np.sort(note_array, order='onset_beat')
        intervals = np.diff(sorted_notes['pitch'])
        # Map intervals to -12 to 12 range, plus "larger" buckets
        interval_dist = np.histogram(intervals, bins=np.arange(-13, 15))[0] / (len(intervals) if len(intervals) > 0 else 1)

        # 4. Global Stats
        avg_pitch = np.mean(pitches)
        std_pitch = np.std(pitches)
        avg_dur = np.mean(durations)
        polyphony = len(note_array) / (np.max(note_array['onset_beat']) - np.min(note_array['onset_beat'])) if len(note_array) > 1 else 0

        features = np.concatenate([
            pitch_dist,
            dur_dist,
            interval_dist,
            [avg_pitch, std_pitch, avg_dur, polyphony]
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {fn}: {e}")
        return None

def load_dataset(datadir: str, label_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(label_file)
    features_list = []
    labels_list = []
    file_names = []

    for _, row in df.iterrows():
        fn = os.path.join(datadir, row['Score'])
        feat = extract_features(fn)
        if feat is not None:
            features_list.append(feat)
            if 'Composer' in df.columns:
                labels_list.append(COMPOSER_CLASSES[row['Composer']])
            else:
                labels_list.append(-1)
            file_names.append(row['Score'])

    return np.array(features_list), np.array(labels_list), file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composer Identification Challenge")
    parser.add_argument("-i", "--input", required=True, help="Path to scores directory")
    parser.add_argument("-o", "--output", required=True, help="Output results CSV")
    parser.add_argument("--model", help="Path to pre-trained model", default=None)
    args = parser.parse_args()

    # Define paths based on project structure
    # The script is expected to be run from the root of the challenge folder
    
    # Attempt to find the CSV files in a few likely locations
    possible_dirs = [
        "Dataset-20260125",
        os.path.dirname(args.input),
        "."
    ]
    
    train_csv = None
    test_csv = None
    
    for d in possible_dirs:
        t_path = os.path.join(d, "composer_classification_training.csv")
        te_path = os.path.join(d, "composer_classification_test_no_labels.csv")
        if os.path.exists(t_path) and train_csv is None:
            train_csv = t_path
        if os.path.exists(te_path) and test_csv is None:
            test_csv = te_path

    model_fn = "bach_model.joblib"

    if os.path.exists(model_fn):
        print(f"Loading pre-trained model from {model_fn}...")
        clf = joblib.load(model_fn)
    else:
        if train_csv is None or not os.path.exists(train_csv):
            raise FileNotFoundError("Training CSV not found and no pre-trained model available.")
        
        print(f"Using training file: {train_csv}")
        print("Loading training data...")
        X_train, y_train, _ = load_dataset(args.input, train_csv)

        print("Training model...")
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        clf.fit(X_train, y_train)
        print(f"Saving model to {model_fn}...")
        joblib.dump(clf, model_fn)

    print("Loading test data...")
    # For the test data, we only have the scores in the input directory
    # If the test_csv is present, we use it to get the filenames
    # If not, we might need to list files, but the challenge provides a csv structure
    
    if test_csv and os.path.exists(test_csv):
        print(f"Using test file list: {test_csv}")
        X_test, _, test_files = load_dataset(args.input, test_csv)
    else:
        print("No test CSV found, attempting to process all .musicxml files in input directory...")
        test_files = [f for f in os.listdir(args.input) if f.endswith(".musicxml")]
        features_list = []
        final_test_files = []
        for f in test_files:
            feat = extract_features(os.path.join(args.input, f))
            if feat is not None:
                features_list.append(feat)
                final_test_files.append(f)
        X_test = np.array(features_list)
        test_files = final_test_files

    print("Predicting...")
    preds = clf.predict(X_test)
    pred_composers = [CLASSES_COMPOSER[p] for p in preds]

    results_df = pd.DataFrame({
        "Score": test_files,
        "Composer": pred_composers
    })

    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

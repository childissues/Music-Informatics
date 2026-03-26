import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from features import extract_features_from_score
from concurrent.futures import ProcessPoolExecutor
import tqdm

def extract_features_wrapper(args):
    file_path, filename = args
    features = extract_features_from_score(file_path)
    if features:
        features['file'] = filename
    return features

def main():
    # Paths
    dataset_dir = "Dataset-20260125"
    scores_dir = os.path.join(dataset_dir, "scores_difficulty_estimation")
    train_csv_path = os.path.join(dataset_dir, "difficulty_classification_training.csv")
    
    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(train_csv_path)
    
    # Check if features already extracted
    features_csv_path = "features_train.csv"
    if os.path.exists(features_csv_path):
        print("Loading features from cache...")
        features_df = pd.read_csv(features_csv_path)
    else:
        print("Extracting features...")
        # Prepare arguments for parallel processing
        tasks = []
        for filename in labels_df['file']:
            file_path = os.path.join(scores_dir, filename)
            tasks.append((file_path, filename))
        
        # Run extraction (parallel usually fails in some envs, let's try sequential first for safety or simple parallel)
        # Using sequential for safety in this environment as multiprocessing inside tools can be tricky with output capture
        results = []
        for task in tqdm.tqdm(tasks):
            results.append(extract_features_wrapper(task))
            
        features_df = pd.DataFrame([r for r in results if r is not None])
        features_df.to_csv(features_csv_path, index=False)
        print(f"Features saved to {features_csv_path}")

    # Merge features with labels
    data = pd.merge(labels_df, features_df, on='file', how='inner')
    print(f"Total samples with features: {len(data)}")
    
    # Prepare X and y
    feature_cols = [c for c in data.columns if c not in ['file', 'difficulty']]
    X = data[feature_cols]
    y = data['difficulty']
    
    # Train Model
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Review Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    
    print(f"CV Accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"CV F1-Macro: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # Final Fit
    clf.fit(X, y)
    
    # Save Model and Columns
    joblib.dump(clf, "model.joblib")
    with open("feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
        
    print("Model and feature columns saved.")

if __name__ == "__main__":
    main()

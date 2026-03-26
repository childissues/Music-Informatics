import warnings
# Suppress all warnings including urllib3 NotOpenSSLWarning
warnings.filterwarnings("ignore")

import argparse
import os
import pandas as pd
import joblib
import json
from features import extract_features_from_score
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Piece Difficulty Estimation - Team J. S. Bach")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to the directory containing MusicXML scores.")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_file = args.output_file
    
    # Load model and feature columns
    # We assume they are in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.joblib")
    features_cols_path = os.path.join(script_dir, "feature_columns.json")
    
    try:
        model = joblib.load(model_path)
        with open(features_cols_path, "r") as f:
            feature_cols = json.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Find all musicxml files
    files = [f for f in os.listdir(input_dir) if f.endswith(".musicxml")]
    if not files:
        print(f"No .musicxml files found in {input_dir}")
        return
        
    print(f"Found {len(files)} files. Processing...")
    
    results = []
    
    # Iterate and predict
    # Note: Sequential processing to avoid multiprocessing issues in simple scripts, 
    # but could be parallelized for speed if needed.
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        
        # Extract features
        features = extract_features_from_score(file_path)
        
        if features:
            # Prepare DataFrame for prediction
            df = pd.DataFrame([features])
            
            # Ensure columns matching
            # Add missing columns with 0
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = 0
            
            # Reorder
            df = df[feature_cols]
            
            # Predict
            pred = model.predict(df)[0]
            
        else:
            # Fallback if extraction fails
            # Default to median difficulty (e.g. 5)
            pred = 5
            
        results.append({"file": filename, "difficulty": int(pred)})
        
        # Simple progress indicator
        if len(results) % 20 == 0:
            print(f"Processed {len(results)}/{len(files)}")
            
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

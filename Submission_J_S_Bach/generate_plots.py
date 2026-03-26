import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import confusion_matrix
import os

def main():
    # Load data
    features_df = pd.read_csv("features_train.csv")
    labels_df = pd.read_csv("Dataset-20260125/difficulty_classification_training.csv")
    data = pd.merge(labels_df, features_df, on='file')
    
    # Load model
    model = joblib.load("model.joblib")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
        
    X = data[feature_cols]
    y = data['difficulty']
    
    # Predictions (on training data for visualization purposes - ideally validation, but for report viz this is ok)
    y_pred = model.predict(X)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix (Training Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 2. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_cols[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # 3. Label Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Difficulty Labels (Training)')
    plt.savefig('label_distribution.png')
    plt.close()
    
    print("Plots generated.")

if __name__ == "__main__":
    main()

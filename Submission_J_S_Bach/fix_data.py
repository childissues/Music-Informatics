import pandas as pd
import numpy as np
from fractions import Fraction

def convert_to_float(x):
    if isinstance(x, str) and '/' in x:
        try:
            return float(Fraction(x))
        except:
            return x
    return x

def main():
    df = pd.read_csv("features_train.csv")
    print("Columns before:", df.dtypes)
    
    # Iterate over all columns and apply conversion
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'file':
            print(f"Fixing column {col}...")
            df[col] = df[col].apply(convert_to_float)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df.to_csv("features_train.csv", index=False)
    print("Fixed CSV saved.")
    print("Columns after:", df.dtypes)

if __name__ == "__main__":
    main()

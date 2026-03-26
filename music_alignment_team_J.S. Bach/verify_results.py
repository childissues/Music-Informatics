
import numpy as np
import pandas as pd
import os

def verify():
    output_file = 'outputs/JSBach_Alignment.npz'
    if not os.path.exists(output_file):
        print("Output file not found!")
        return

    data = np.load(output_file, allow_pickle=True)
    
    # Loop over all pieces
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    piece_scores = []
    
    for piece_name in data:
        # Load GT
        gt_path = f'trainingset/{piece_name}/align.csv'
        if not os.path.exists(gt_path):
            continue
            
        gt_df = pd.read_csv(gt_path)
        
        gt_pairs = set()
        for _, row in gt_df.iterrows():
            if str(row['matchtype']) == '0':
                gt_pairs.add((str(row['partid']), str(row['ppartid'])))
                
        my_alignment = data[piece_name]
        my_pairs = set()
        for row in my_alignment:
            my_pairs.add((str(row[0]), str(row[1])))
            
        tp = len(gt_pairs.intersection(my_pairs))
        fp = len(my_pairs - gt_pairs)
        fn = len(gt_pairs - my_pairs)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        piece_scores.append(f1)
        
    # Global Micro Average
    precision_micro = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall_micro = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    
    # Global Macro Average
    f1_macro = np.mean(piece_scores)
    
    print(f"Global Evaluation on {len(piece_scores)} pieces")
    print(f"Total Correct Matches: {all_tp}")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    print(f"Micro F1 Score: {f1_micro:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")

if __name__ == "__main__":
    verify()

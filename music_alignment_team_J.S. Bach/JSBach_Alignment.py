
import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import numba

@numba.jit(nopython=True)
def compute_accumulated_cost_matrix(C):
    N, M = C.shape
    D = np.zeros((N + 1, M + 1))
    D[0, 0] = 0
    D[1:, 0] = np.inf
    D[0, 1:] = np.inf
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = C[i-1, j-1]
            # Standard DTW step: min of (match, insertion, deletion)
            val = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            D[i, j] = cost + val
            
    return D

@numba.jit(nopython=True)
def dtw_backtrace(D):
    N = D.shape[0] - 1
    M = D.shape[1] - 1
    i, j = N, M
    path = []
    
    # We want to trace back to (1, 1), which corresponds to (0,0) in our index inputs
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        if i == 1 and j == 1:
            break
            
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            # 0: vertical (i-1, j), 1: horizontal (i, j-1), 2: diagonal (i-1, j-1)
            # D indices:
            d_diag = D[i-1, j-1]
            d_vert = D[i-1, j]
            d_horz = D[i, j-1]
            
            # Prefer diagonal if equal? Usually yes.
            if d_diag <= d_vert and d_diag <= d_horz:
                i -= 1
                j -= 1
            elif d_vert <= d_horz: # prefer vertical (deletion) over horizontal?
                i -= 1
            else:
                j -= 1

    # Reverse path to get start-to-end
    return path[::-1] # List of tuples

def load_data(piece_path):
    part_df = pd.read_csv(os.path.join(piece_path, 'part.csv'))
    ppart_df = pd.read_csv(os.path.join(piece_path, 'ppart.csv'))
    return part_df, ppart_df

def align_piece(piece_path):
    part, ppart = load_data(piece_path)
    
    # Sort to ensure monotonic time order, and canonical pitch order for chords
    # Score: Sort by Beat then Pitch (Canonical)
    part = part.sort_values(by=['onset_beat', 'pitch'])
    
    # Perf: Sort by Quantized Time then Pitch (to fix arpeggio crossings)
    # 0.2s window allows reordering of notes played within ~200ms
    ppart['onset_quant'] = (ppart['onset_sec'] // 0.2)
    ppart = ppart.sort_values(by=['onset_quant', 'pitch'])
    
    # features
    score_pitch = part['pitch'].values.astype(float)
    perf_pitch = ppart['pitch'].values.astype(float)
    
    # Cost Matrix
    C = cdist(score_pitch.reshape(-1, 1), perf_pitch.reshape(-1, 1), metric='cityblock')
    
    # DTW
    D = compute_accumulated_cost_matrix(C)
    path = dtw_backtrace(D)
    
    # Convert path indices to IDs
    # path is list of (part_idx, ppart_idx)
    path = np.array(path)
    
    part_ids = part.iloc[path[:, 0]]['id'].astype(str).values
    ppart_ids = ppart.iloc[path[:, 1]]['id'].astype(str).values
    
    # Stack them
    alignment_result = np.column_stack((part_ids, ppart_ids)).astype(str)
    return alignment_result

def main():
    parser = argparse.ArgumentParser(description="J. S. Bach Alignment Script")
    parser.add_argument('-c', action='store_true', help="Challenge mode")
    parser.add_argument('-i', '--input_dir', default='trainingset', help="Input directory containing dataset")
    parser.add_argument('-o', '--output_dir', default='outputs', help="Output directory for results")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find all pieces
    # Structure: input_dir/PieceName/part.csv
    # So we list subdirectories
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    subdirs.sort()
    
    all_results = {}
    
    for piece_name in subdirs:
        piece_path = os.path.join(input_dir, piece_name)
        # Check if csvs exist
        if not (os.path.exists(os.path.join(piece_path, 'part.csv')) and \
                os.path.exists(os.path.join(piece_path, 'ppart.csv'))):
            continue
            
        print(f"Aligning {piece_name}...")
        alignment = align_piece(piece_path)
        all_results[piece_name] = alignment
        
    # Save results
    output_files = os.path.join(output_dir, "JSBach_Alignment.npz")
    np.savez_compressed(output_files, **all_results)
    print(f"Computed alignments for {len(all_results)} pieces.")
    print(f"Saved results to {output_files}")

if __name__ == "__main__":
    main()

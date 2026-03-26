import warnings
# Suppress all warnings including urllib3 NotOpenSSLWarning
warnings.filterwarnings("ignore")

import music21
import numpy as np
import pandas as pd
import os

def extract_features_from_score(file_path):
    """
    Extracts features from a MusicXML file using music21.
    """
    try:
        score = music21.converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    features = {}
    
    try:
        # Flatten the score to get all elements in chronological order
        flat_score = score.flat
        
        # Filter for Notes and Chords
        notes_and_chords = list(flat_score.notes)
        
        if not notes_and_chords:
            return None

        # 1. Pitch Features
        pitches = []
        for nc in notes_and_chords:
            if isinstance(nc, music21.note.Note):
                pitches.append(nc.pitch.ps)
            elif isinstance(nc, music21.chord.Chord):
                for p in nc.pitches:
                    pitches.append(p.ps)
        
        pitches = np.array(pitches)
        if len(pitches) > 0:
            features['pitch_min'] = np.min(pitches)
            features['pitch_max'] = np.max(pitches)
            features['pitch_range'] = np.max(pitches) - np.min(pitches)
            features['pitch_mean'] = np.mean(pitches)
            features['pitch_std'] = np.std(pitches)
        else:
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_range'] = 0
            features['pitch_mean'] = 0
            features['pitch_std'] = 0

        # 2. Rhythm/Duration Features
        durations = [nc.duration.quarterLength for nc in notes_and_chords]
        durations = np.array(durations)
        
        features['duration_total'] = float(score.duration.quarterLength)
        features['n_notes_total'] = len(pitches) # Total note heads
        features['n_events'] = len(notes_and_chords) # Notes + Chords events
        
        if features['duration_total'] > 0:
            features['note_density'] = features['n_notes_total'] / features['duration_total']
            features['event_density'] = features['n_events'] / features['duration_total']
        else:
            features['note_density'] = 0
            features['event_density'] = 0
            
        if len(durations) > 0:
            features['duration_mean'] = np.mean(durations)
            features['duration_std'] = np.std(durations)
            features['duration_min'] = np.min(durations)
        else:
            features['duration_mean'] = 0
            features['duration_std'] = 0
            features['duration_min'] = 0

        # 3. Chord/Polyphony Features
        chord_sizes = []
        for nc in notes_and_chords:
            if isinstance(nc, music21.chord.Chord):
                chord_sizes.append(len(nc.notes))
            else:
                chord_sizes.append(1)
        
        features['chord_size_mean'] = np.mean(chord_sizes)
        features['chord_size_max'] = np.max(chord_sizes)
        
        # 4. Interval Features (Complexity)
        # Calculate intervals between consecutive events (melody-like)
        # This is a simplification; ideally we'd separate voices, but flattening mixes them.
        # Still, large jumps in the flattened stream often indicate complexity or wide hand movement.
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            features['interval_mean'] = np.mean(np.abs(intervals))
            features['interval_std'] = np.std(np.abs(intervals))
            features['interval_max'] = np.max(np.abs(intervals))
        else:
            features['interval_mean'] = 0
            features['interval_std'] = 0
            features['interval_max'] = 0
            
        # 5. Key/Scale Features (Expensive, maybe optional)
        # features['n_accidentals'] = sum([1 for p in pitches if not music21.pitch.Pitch(p).simplifyEnharmonic().step in ['C','D','E','F','G','A','B']]) # Very rough approximation
        
        # Using music21 key analysis (might be slow)
        # k = score.analyze('key')
        # features['key_fifths'] = k.sharps # Positive for sharps, negative for flats

    except Exception as e:
        print(f"Error processing features for {file_path}: {e}")
        return None

    return features

if __name__ == "__main__":
    # Test on a dummy file if it exists, or just print message
    print("Feature extractor module ready.")

# Music Generation — Team J.S. Bach (Python Code)

This submission contains the Python code to generate our composed piece for the Music Generation task.

## Files
- `Team_JS_Bach_MusicGeneration.py` — main script to generate the MIDI
- `data/piece_spec.json` — compact symbolic specification (parameters) used by the generator
- `environment.yml` — conda environment specification

## Setup
```bash
conda env create -f environment.yml
conda activate musicgen
```

## Run
From the unzipped folder:
```bash
python Team_JS_Bach_MusicGeneration.py --out_dir output
```

Output:
- `output/Team_JS_Bach_generated.mid`

## Optional controls
- Transpose: `--transpose 12`
- Softer/louder: `--velocity_scale 0.9`
- Faster/slower: `--tempo_scale 1.05`

## Trained model parameters
This project does not use a trained model, so there are no model weights to include.

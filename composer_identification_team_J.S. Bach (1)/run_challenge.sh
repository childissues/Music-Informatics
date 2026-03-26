#!/bin/bash

# Activate the virtual environment if it exists in the standard location
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

# Run the composer identification script with the correct arguments
# We use quotes to handle potential spaces in paths
python "J. S. Bach_ComposerIdentification.py" -i "Dataset-20260125/scores_composer_identification" -o "results_composer_identification.csv"

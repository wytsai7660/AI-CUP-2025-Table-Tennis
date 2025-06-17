#!/bin/bash

set -e

cd Gender
bash download_dataset.sh
bash make_prediction.sh
echo -e "Gender prediction completed."
cd ..

cd "Hold Racket Hand"
bash download_dataset.sh
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/classifier.pkl
python test.py
echo -e "Hold Racket Hand prediction completed."
cd ..

cd Level
bash download_dataset.sh
python3 test.py
echo -e "Level prediction completed."
cd ..

cd Play_years
bash download_dataset.sh
bash make_prediction.sh
echo -e "Play_years prediction completed."
cd ..

python3 merge_predictions.py
echo -e "All predictions merged successfully."
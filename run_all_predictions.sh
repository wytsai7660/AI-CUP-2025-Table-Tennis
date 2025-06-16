#!/bin/bash

set -e

cd Gender
bash make_prediction.sh
echo -e "Gender prediction completed."
cd ..

cd "Hold Racket Hand"
bash make_prediction.sh
echo -e "Hold Racket Hand prediction completed."
cd ..

cd Level
bash download_dataset.sh
python3 test.py
echo -e "Level prediction completed."
cd ..

cd Play_years
bash make_prediction.sh
echo -e "Play_years prediction completed."
cd ..

python3 merge_predictions.py
echo -e "All predictions merged successfully."
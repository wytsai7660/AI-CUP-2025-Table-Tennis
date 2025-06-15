#!/bin/bash
# download model and data
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/weights.zip
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/39_Training_Dataset.zip
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/39_Test_Dataset.zip
unzip -q weights.zip -d ./
unzip -q 39_Training_Dataset.zip -d ./
unzip -q 39_Test_Dataset.zip -d ./
rm -f 39_Training_Dataset.zip
rm -f 39_Test_Dataset.zip
rm -f weights.zip

# run the prediction script
python3 test.py

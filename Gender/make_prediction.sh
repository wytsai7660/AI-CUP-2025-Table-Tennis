#!/bin/bash
# download model
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/weights.zip
unzip -q weights.zip -d ./
rm -f weights.zip

# run the prediction script
python3 test.py

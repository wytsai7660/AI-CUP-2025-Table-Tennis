#!/bin/bash
# download model
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/imugpt_weights.zip
unzip -q imugpt_weights.zip -d ./
rm -f imugpt_weights.zip

# run the prediction script
python3 test.py

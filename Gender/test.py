import torch
import torch.nn as nn
import numpy as np
import csv
from encoder_model import Configs, SwingGRU

configs = Configs()
configs.pos_embed = False

from dataloader_d import TrajectoryDataset
from config import TEST_INFO, TEST_DATA_DIR, TRAIN_INFO, TRAIN_DATA_DIR
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(TRAIN_INFO)
test_df = pd.read_csv(TEST_INFO)

PREDICTING_FIELDS = [
    "mode",
    "gender",
    "hold racket handed",
    # "play years",
    "level",
]

TEST_PREDICTING_FIELDS = [
    "mode",
]

import joblib
scaler = joblib.load(f"weights/scaler.pkl")
encoders = joblib.load(f"weights/encoders.pkl")


mode_2_model_gender = [
    "weights/gender/epoch_7_loss_1.5972_f1_0.8495_roc_nan_mode_1.pth",
    "weights/gender/epoch_18_loss_2.2859_f1_0.8495_roc_nan_mode_2.pth",
    "weights/gender/epoch_20_loss_0.6645_f1_1.0000_roc_nan_mode_3.pth",
    "weights/gender/epoch_1_loss_0.8701_f1_1.0000_roc_nan_mode_4.pth",
    "weights/gender/epoch_17_loss_0.5895_f1_1.0000_roc_nan_mode_5.pth",
    "weights/gender/epoch_13_loss_2.2302_f1_0.9836_roc_nan_mode_6.pth",
    "weights/gender/epoch_19_loss_0.4666_f1_1.0000_roc_nan_mode_7.pth",
    "weights/gender/epoch_17_loss_1.0645_f1_0.9699_roc_nan_mode_8.pth",
    "weights/gender/epoch_3_loss_1.1968_f1_0.9491_roc_nan_mode_9.pth",
    "weights/gender/epoch_19_loss_0.7044_f1_1.0000_roc_nan_mode_10.pth",
]


all_results = {}

for i, mode in enumerate(range(1, 11)):
    print(f"Loading model for mode {mode}, path: {mode_2_model_gender[i]}")
    

    model = SwingGRU(configs)
    model.load_state_dict(torch.load(mode_2_model_gender[i], map_location=device, weights_only=True))
    model = model.to(device)

    class_nums = [len(encoder.classes_) for encoder in encoders]
    print("Class numbers for each label:", class_nums)
    cumsum_class_nums = np.cumsum([0] + class_nums)
    task_weight = [1.0, 1.0, 1.0, 1.0, 1.0]  # Example weights for each task

    criterions = [(s, e, nn.CrossEntropyLoss(), weight) for s, e, weight in zip(cumsum_class_nums[:-1], cumsum_class_nums[1:], task_weight)]
    
    
    test_dataset = TrajectoryDataset(
        TEST_DATA_DIR,
        test_df[test_df["mode"] == mode],
        max_seq_len=configs.seq_len,
        label=False,
        predicting_fields=TEST_PREDICTING_FIELDS,
        scaler=scaler,
        encoder=encoders,
        augmentation=False,
        sample_weights=False,
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    with torch.no_grad():
        for i, (sample, id) in enumerate(zip(test_dataset.samples, test_dataset.unique_ids)):
            data, _ = sample
            if data.shape[0] < configs.seq_len:
                data = np.pad(data, ((0, configs.seq_len - data.shape[0]), (0, 0)), mode='constant', constant_values=0)

            data = scaler.transform(data)
            patches = []
            for start in range(0, len(data) - configs.seq_len + 1, configs.patch_size * 4):
                end = start + configs.seq_len
                if end > len(data):
                    break
                patch = data[start:end]
                patch = torch.tensor(patch, dtype=torch.float32)
                patches.append(patch)
            
            patches = torch.stack(patches, dim=0).to(device)
            print(f"number of patches: {len(patches)}")
            
            probs = model(patches, criterions=criterions)
            probs = torch.mean(probs[:, 10:12], dim=0).cpu().numpy()
            all_results[id] = probs
            
    sorted_results = sorted(all_results.items(), key=lambda x: x[0])

    with open("gender.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        required_columns = [
            "unique_id", "gender", "hold racket handed", "play years_0",
            "play years_1", "play years_2", "level_2", "level_3", "level_4", "level_5"
        ]
        writer.writerow(required_columns)

        for id, probs in sorted_results:
            rounded_probs = round(probs[0], 4)
            writer.writerow([id] + [rounded_probs] + [0.5] * (len(required_columns) - 2))

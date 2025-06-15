#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import TRAIN_DATA_DIR, TRAIN_INFO, TEST_INFO, TEST_DATA_DIR
from tqdm import tqdm
from gpt import GPT, GPTConfig
from dataloader import TrajectoryDataset
import matplotlib.pyplot as plt
import joblib
import os
import wandb
import glob
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

def svm_evaluate(model, dataset):
    all_embeddings = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    with torch.no_grad():
        for input, _, label in tqdm(dataloader, desc="Extracting SVM embeddings", leave=False):
            input, label_cu = input.to(device), label.to(device)
            if model_args.enable_mode_embedding:
                embedding = model(input, mode=label_cu[:, 0])
            else:
                embedding = model(input)
            embedding = torch.mean(embedding, dim=1).detach().cpu().numpy()
            all_embeddings.append(embedding)
            all_labels.append(label.numpy())
    
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    print(f"all_embeddings shape: {embeddings.shape}")
    print(f"all_labels shape: {labels.shape}")
        
    df = pd.read_csv(TRAIN_INFO)
    unique_players = sorted(df["player_id"].unique())
    
    for task in PREDICTING_FIELDS[1:]:
        task_labels = labels[:, PREDICTING_FIELDS.index(task)]
        _, label_count = np.unique(task_labels, return_counts=True)
        print(f"Task: {task}, Unique labels: {label_count}")
        task_labels = task_labels.astype(int)
        unique_labels = np.unique(task_labels)
        
        all_predictions = []
        all_predictions_probs = []
        all_ground_truths = []
        print(f"Evaluating SVM for task: {task}")
        for valid_player_ids in tqdm(unique_players, desc="Validating players", leave=False):
            # print(f"player: {valid_player_ids}")
            train_player_ids = [p for p in unique_players if p != valid_player_ids]
            df_train = df[df["player_id"].isin(train_player_ids)].index.values
            df_valid = df[(df["player_id"] == valid_player_ids)].index.values

            train_embeddings = embeddings[df_train]
            train_labels = task_labels[df_train]
            valid_embeddings = embeddings[df_valid]
            valid_labels = task_labels[df_valid]
            
            # Train SVM
            svm = SVC(C=1.0, probability=True, kernel='rbf', class_weight='balanced')
            svm.fit(train_embeddings, train_labels)
            
            # Predict
            predictions = svm.predict(valid_embeddings)
            probabilities = svm.predict_proba(valid_embeddings)
            
            all_predictions.append(predictions)
            all_predictions_probs.append(probabilities)
            all_ground_truths.append(valid_labels)
            
            joblib.dump(svm, f"{out_dir}/svm_{task}_{valid_player_ids}.joblib")
        
        all_predictions = np.concatenate(all_predictions)
        all_predictions_probs = np.concatenate(all_predictions_probs)
        all_ground_truths = np.concatenate(all_ground_truths)
        
        print(f"Classification Report for {task}:")
        print(classification_report(all_ground_truths, all_predictions))
        f1 = f1_score(all_ground_truths, all_predictions, average='macro')
        accuracy = accuracy_score(all_ground_truths, all_predictions)
        print(f"F1 Score for {task}: {f1:.4f}")
        print(f"Accuracy for {task}: {accuracy:.4f}")
        if len(unique_labels) == 2:
            micro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs[:, 1], average='micro')
            macro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs[:, 1], average='macro')
            
            print(f"Micro ROC AUC score: {micro_roc_auc:.4f}")
            print(f"Macro ROC AUC score: {macro_roc_auc:.4f}")
        else:
            micro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs, average='micro', multi_class='ovr')
            macro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs, average='macro', multi_class='ovr')

            print(f"Micro ROC AUC score: {micro_roc_auc:.4f}")
            print(f"Macro ROC AUC score: {macro_roc_auc:.4f}")
            
        if use_wandb:
            wandb.log({
                f"{task}_f1_score": f1,
                f"{task}_accuracy": accuracy,
                f"{task}_micro_roc_auc": micro_roc_auc,
                f"{task}_macro_roc_auc": macro_roc_auc,
            })
#%%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_args = GPTConfig(
    max_seq_len = 1024,
    in_chans = 2,
    n_layer = 8,
    n_head = 8,
    n_embd = 128,
    ff = 4,
    patch_size = 8,
    dropout = 0.2,
    bias = False,
    enable_mode_embedding = False,
    mlp_prediction_head = True,
)

batch_size = 32
learning_rate = 5e-3
weight_decay = 0.001
betas = (0.9, 0.95)
num_epochs = 5000

use_scaler = True
detrend_and_filter = True
data_transform = "magnitude_only"
out_dir = "outs/out70"
use_wandb = False

PREDICTING_FIELDS = [
    "mode",
    # "gender",
    # "hold racket handed",
    "play years",
    # "level",
]

TEST_PREDICTING_FIELDS = [
    "mode",
]

os.makedirs(out_dir, exist_ok=True)

print(f"preparing dataset...")
train_df = pd.read_csv(TRAIN_INFO)
test_df = pd.read_csv(TEST_INFO)

# scaler = joblib.load("scaler.joblib") if use_scaler else None
#%%
train_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    train_df,
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    scaler=None,
    detrend_and_filter=detrend_and_filter,
    sample_weights=False,
    label=True,
    predicting_fields=PREDICTING_FIELDS,
    data_transform=data_transform,
)

if use_scaler:
    joblib.dump(train_dataset.scaler, f"{out_dir}/scaler_year.joblib")

#%%
model = GPT(model_args)
model.load_state_dict(torch.load(f"{out_dir}/model_epoch_205.pth", map_location="cpu"))
model = model.to(device)
model.eval()

svm_years_models = glob.glob(f"{out_dir}/svm_*.joblib")
if not svm_years_models:
    print("No SVM models found in output directory. Running SVM evaluation...")
    svm_evaluate(model, train_dataset)
else:
    print(f"Found {len(svm_years_models)} SVM models in output directory. Skipping SVM evaluation.")

#%%
svm_dataset = TrajectoryDataset(
    TEST_DATA_DIR,
    test_df,
    train=False,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    scaler=train_dataset.scaler if use_scaler else None,
    encoder=train_dataset.encoders,
    sample_weights=False,
    detrend_and_filter=detrend_and_filter,
    # label=True,
    predicting_fields=TEST_PREDICTING_FIELDS,
    data_transform=data_transform,
)
# dataloader = DataLoader(svm_dataset, batch_size=128, shuffle=False, pin_memory=True)

#%%
fpaths = [
    TEST_DATA_DIR / f"{unique_id}.txt" for unique_id in test_df["unique_id"].values
]
#%%
from segment import Trim
from scipy.signal import detrend
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)
trim_method = Trim()
#%%
from pathlib import Path

svm_years_models = glob.glob(f"{out_dir}/svm_play years_*.joblib")

svm_models = {}
for model_path in svm_years_models:
    model_name = Path(model_path).stem
    svm_models[model_name] = joblib.load(model_path)

#%%

all_level_probs = []
# all_year_probs = []
for i, fpath in enumerate(fpaths):
    data = np.loadtxt(fpath)
    data = torch.tensor(detrend(data, axis=0), dtype=torch.float32)
    data = trim_method(data)[0].numpy()
    data = butter_lowpass_filter(data, cutoff=30, fs=85, order=4)
    data = data.astype(np.float32)
    acc_mag = np.linalg.norm(data[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    data = np.concatenate((acc_mag, gyro_mag), axis=1)

    if data.shape[0] < model_args.max_seq_len:
        data = np.pad(data, ((0, model_args.max_seq_len - data.shape[0]), (0, 0)), mode='constant', constant_values=0)

    data = train_dataset.scaler.transform(data)
    
    patches = []
    for start in range(0, len(data) - model_args.max_seq_len + 1, model_args.patch_size):
        end = start + model_args.max_seq_len
        if end > len(data):
            break
        patch = data[start:end]
        patch = torch.tensor(patch, dtype=torch.float32)
        patches.append(patch)
    
    patches = torch.stack(patches, dim=0).to(device)
    print(f"Loaded {fpath}, number of patches: {len(patches)}")
        
    with torch.no_grad():
        embedding = model(patches)
        embedding = torch.mean(embedding, dim=1).detach().cpu().numpy()
    
    print(f"Extracted embedding for {fpath}, shape: {embedding.shape}")
    
    all_probs = []
    for model_name, svm_model in svm_models.items():
        probabilities = svm_model.predict_proba(embedding)
        all_probs.append(probabilities)
    all_probs = np.mean(all_probs, axis=0)
    level_prob = np.mean(all_probs, axis=0)
    print(level_prob)
    all_level_probs.append(level_prob)

#%%
unique_ids = test_df["unique_id"].values.astype(int)
# 四捨五入到小數點後4位
all_level_probs_round = np.round(np.array(all_level_probs), 4)

# 增加5個other_i欄位，內容填0.5
other_columns = [f'other_{i}' for i in range(5)]
other_data = np.full((all_level_probs_round.shape[0], 5), 0.5)

# level_columns = ['level_2', 'level_3', 'level_4', 'level_5']
level_columns = ['play years_0', 'play years_1', 'play years_2']
# 合併所有欄位
df_out = pd.DataFrame(
    np.concatenate([unique_ids[:, None], all_level_probs_round], axis=1),
    columns=['unique_id'] + level_columns
)
df_out['unique_id'] = df_out['unique_id'].astype(int)
df_out.to_csv("temp.csv", index=False, float_format="%.4f")

#%%

import pandas as pd

required_columns = [
    "unique_id", "gender", "hold racket handed", "play years_0",
    "play years_1", "play years_2", "level_2", "level_3", "level_4", "level_5"
]

df = pd.read_csv("temp.csv")

for col in required_columns:
    if col not in df.columns:
        df[col] = 0.5

df = df[required_columns]
df.to_csv("play_years.csv", index=False)
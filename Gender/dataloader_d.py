import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from segment import Trim
from scipy.signal import detrend
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

def add_gaussian_noise(data, noise_level=0.01):
    sigma = noise_level * np.std(data)
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def scale_magnitude(data, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale_factor

def random_masking(data, mask_ratio=0.1, mask_value=0.0):
    T = data.shape[0]
    augmented_data = data.copy()
    mask_len = int(T * mask_ratio)
    if mask_len == 0:
        return augmented_data

    start_idx = np.random.randint(0, T - mask_len + 1)

    if isinstance(mask_value, str) and mask_value == 'mean':
        fill_values = np.mean(data, axis=0) # 計算每列的均值
        augmented_data[start_idx : start_idx + mask_len, :] = np.tile(fill_values, (mask_len, 1))
    else:
        augmented_data[start_idx : start_idx + mask_len, :] = mask_value

    return augmented_data

aug_funcs = [add_gaussian_noise, scale_magnitude, random_masking]

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        data_dir: Path,
        dataframe: pd.DataFrame,
        max_seq_len: int = 128,
        train: bool = True,
        label: bool = False,
        detrend_and_filter: bool = True,
        scaler: StandardScaler = None,
        encoder: List[LabelEncoder] = None,
        predicting_fields: List[str] = None,
        sample_weights: bool = False,
        freq_cutoff: int = 40,
        filter_order: int = 2,
        augmentation: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.label = label
        self.encoders = encoder
        self.augmentation = augmentation
        trim_method = Trim()

        if label:
            df_to_encode = dataframe[predicting_fields]
            if self.encoders is None:
                self.encoders = [LabelEncoder().fit(df_to_encode[col]) for col in df_to_encode.columns]
            encoded = [enc.transform(df_to_encode[col])[:, None] for enc, col in zip(self.encoders, df_to_encode.columns)]
            encoded = np.concatenate(encoded, axis=1)
            metas = torch.tensor(encoded, dtype=torch.long)
            
            if sample_weights:
                cal_metas = metas[:, 1:]  # Exclude the first column (unique_id)
                # cal_metas = metas
                unique_combinations, counts = np.unique(cal_metas.numpy(), axis=0, return_counts=True)
                weights = max(counts) / counts
                weights = np.round(weights).astype(int)
                comb2weight = {tuple(comb): w for comb, w in zip(unique_combinations, weights)}
                sample_weights = np.array([comb2weight[tuple(meta.tolist())] for meta in cal_metas])
            else:
                sample_weights = np.ones((len(dataframe),), dtype=int)
        else:
            # a same len meta but all zeros
            metas = torch.zeros((len(dataframe),), dtype=torch.float32)
            sample_weights = np.ones((len(dataframe),), dtype=int)
            
        # Calculate class weights for each class (for multi-class classification)
        if label:
            class_weights = []
            for i, enc in enumerate(self.encoders):
                labels = metas[:, i].numpy()
                classes, counts = np.unique(labels, return_counts=True)
                weights = np.zeros(len(enc.classes_), dtype=np.float32)
                weights[classes] = len(labels) / (len(classes) * counts)
                class_weights.append(torch.tensor(weights, dtype=torch.float32))
            self.class_weights = class_weights
        else:
            self.class_weights = None
        
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        fpaths = [
            data_dir / f"{unique_id}.txt" for unique_id in dataframe["unique_id"].values
        ]
        self.unique_ids = dataframe["unique_id"].values
        
        all_features = []
        pbar = tqdm(zip(fpaths, metas, sample_weights), total=len(fpaths), desc="Loading data", leave=False)
        
        for fpath, meta, sample_weight in pbar:
            data = np.loadtxt(fpath)
            if detrend_and_filter:
                data = torch.tensor(detrend(data, axis=0), dtype=torch.float32) # trim method will not work with numpy arrays
            else:
                data = torch.tensor(data, dtype=torch.float32)
            data = trim_method(data)[0].numpy()
            # data_diff = np.diff(data, axis=0, prepend=np.zeros((1, data.shape[1])))
            # acc_mag = np.linalg.norm(data_diff[:, 0:3], axis=1, keepdims=True)
            # gyro_mag = np.linalg.norm(data_diff[:, 3:6], axis=1, keepdims=True)
            # acc_mag = np.linalg.norm(data[:, 0:3], axis=1, keepdims=True)
            # gyro_mag = np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
            # data = np.concatenate((acc_mag, gyro_mag), axis=1)
            # data = np.concatenate((data, acc_mag, gyro_mag), axis=1)
            data = self.butter_lowpass_filter(data, cutoff=freq_cutoff, fs=85, order=filter_order)
            data = data.astype(np.float32)
            all_features.append(data)
            
            if train:
                for _ in range(sample_weight):
                    self.samples.append((data, meta))
            else:
                self.samples.append((data, None))
        
        if scaler is None:
            all_features = np.concatenate(all_features)
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
        else:
            self.scaler = scaler
    
    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        seq = item[0]
        
        if self.augmentation:
            for aug_func in aug_funcs:
                if np.random.rand() < 0.5:
                    seq = aug_func(seq)
        
        seq_len_needed = self.max_seq_len
        
        if seq.shape[0] <= seq_len_needed:
            seq = np.pad(seq, ((0, seq_len_needed - seq.shape[0]), (0, 0)), mode='constant', constant_values=0)
            seq = self.scaler.transform(seq).astype(np.float32)
            mask = np.zeros(seq_len_needed, dtype=np.float32)
            mask[:item[0].shape[0]] = 1.0         
            if self.label:
                return seq, mask, item[1].squeeze()   
            return seq, mask

        # Only call randint if the range is valid
        max_start = seq.shape[0] - seq_len_needed
        segment_start = torch.randint(0, max_start, (1,)).item() if max_start > 0 else 0
        seq = self.scaler.transform(seq).astype(np.float32)
        mask = np.ones(self.max_seq_len, dtype=np.float32)
        
        if self.label:
            return seq[segment_start:segment_start + self.max_seq_len], mask, item[1].squeeze()
        return seq[segment_start:segment_start + self.max_seq_len], mask

    
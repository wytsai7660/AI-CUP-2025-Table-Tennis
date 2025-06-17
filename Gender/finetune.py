import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import math
import wandb
from encoder_model import Configs, SwingGRU


if __name__ == "__main__":
    configs = Configs()
    configs.pos_embed = True  # Enable positional embedding
    model = SwingGRU(configs)
    print(model)
    
    from dataloader_d import TrajectoryDataset
    from config import TRAIN_INFO, TRAIN_DATA_DIR
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from copy import deepcopy
    from sklearn.metrics import f1_score, classification_report, roc_auc_score
    import os
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    train_df = pd.read_csv(TRAIN_INFO)
    unique_player = train_df.drop_duplicates(subset=["player_id"])
    
    train_player_ids, val_player_ids = train_test_split(
        unique_player["player_id"].to_numpy(), test_size=0.2, random_state=42, shuffle=True,
        # stratify=unique_player["gender"].to_numpy(),
    )
    
    PREDICTING_FIELDS = [
        "mode",
        # "gender",
        # "hold racket handed",
        "play years",
        # "level",
    ]
    
    import joblib
    scaler = joblib.load(f"weights/scaler.pkl")
    encoders = joblib.load(f"weights/encoders.pkl")

    
    for mode in range(1, 11):
        
        model.load_state_dict(torch.load(f"weights/play years/epoch_25_loss_30.7749_f1_0.6796.pth", map_location=device, weights_only=True))
        model = model.to(device)
        print(f"\n--- Training for mode {mode} ---")
    
        train_dataset = TrajectoryDataset(
            TRAIN_DATA_DIR,
            train_df[train_df["player_id"].isin(train_player_ids) & (train_df["mode"] == mode)],
            max_seq_len=configs.seq_len,
            label=True,
            predicting_fields=PREDICTING_FIELDS,
            # sample_weights=True,
            scaler=scaler,
            encoder=encoders,
        )
                
        valid_dataset = TrajectoryDataset(
            TRAIN_DATA_DIR,
            train_df[train_df["player_id"].isin(val_player_ids) & (train_df["mode"] == mode)],
            max_seq_len=configs.seq_len,
            label=True,
            predicting_fields=PREDICTING_FIELDS,
            scaler=scaler,
            encoder=encoders,
            # sample_weights=True,
            
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
            
        # Get class num for each label from train_dataset.encoders (list of LabelEncoders)
        class_nums = [len(encoder.classes_) for encoder in encoders]
        print("Class numbers for each label:", class_nums)
        cumsum_class_nums = np.cumsum([0] + class_nums)
        task_weight = [0.0, 1.0, 1.0, 1.0, 1.0]  # Example weights for each task
        
        class_weights = train_dataset.class_weights
        print("Class weights for each label:", class_weights)
        criterions = [(s, e, nn.CrossEntropyLoss(weight=c_weight.to(device)), weight) for s, e, weight, c_weight in zip(cumsum_class_nums[:-1], cumsum_class_nums[1:], task_weight, class_weights)]
        print("Criterions:", criterions)
        
        x, mask, target = next(iter(train_loader))
        print(f"\n--- Data Loader Test Run Output ---")
        print(f"Batch shape: {x.shape}")  # Should be [B, T, C]
        print(f"Mask shape: {mask.shape}")  # Should be [B, T]
        print(f"Target shape: {target.shape}")  # Should be [B, T, C]
        # exit(0)

        # Test optimizer configuration (optional)
        optimizer = model.configure_optimizers(weight_decay=1e-3, learning_rate=1e-4)
        print("\nOptimizer configured successfully.")

        
        # Make sure the weight directories for each predicting field exist
        for field in PREDICTING_FIELDS:
            os.makedirs(f"weights/{field}", exist_ok=True)
        
        # model.load_state_dict(torch.load("transformer_best.pth", map_location=device, weights_only=True))
        model = model.to(device)
        
        if configs.use_wandb:
            wandb.init(
                project="swinggru_002",
                config={
                    "seq_len": configs.seq_len,
                    "patch_size": configs.patch_size,
                    "in_chans": configs.in_chans,
                    "n_embd": configs.n_embd,
                    "n_layer": configs.n_layer,
                    "dropout": configs.dropout,
                    "out_size": configs.out_size,
                    "bias": configs.bias,
                    "pos_embed": configs.pos_embed,
                    "predicting_fields": PREDICTING_FIELDS,
                    "sample_weights": PREDICTING_FIELDS[1:],
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-3,
                    "num_epochs": num_epochs,
                    "batch_size": 32,
                    "group_num": 7,
                    "task_weight": task_weight,  # Example weights for each task
                }
            )
            
            
        total_loss = 0.0
        task_losses = np.zeros(len(criterions), dtype=np.float32)
        all_predictions = []
        all_targets = []
        model.eval()
        for _ in range(10):
            for batch_idx, (x_batch, mask_batch, target_batch) in enumerate(valid_loader):
                with torch.no_grad():
                    x_batch, mask_batch, target_batch = x_batch.to(device), mask_batch.to(device), target_batch.to(device)
                    logits, loss, losses = model(x_batch, target=target_batch.long(), criterions=criterions)
                    total_loss += loss.item()
                    task_losses += losses
                    all_predictions.append(logits.cpu().numpy())
                    all_targets.append(target_batch.cpu().numpy())
        
        avg_val_loss = total_loss / len(valid_loader)        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        task_f1 = np.zeros(len(criterions), dtype=np.float32)
        task_roc_auc = np.zeros(len(criterions), dtype=np.float32)
        for i, (s, e, _, _) in enumerate(criterions):
            predictions = np.argmax(all_predictions[:, s:e], axis=-1)
            targets = all_targets[:, i]
            # accuracy = accuracy_score(target, predictions)
            print(f"Task {i} - Loss: {task_losses[i]:.4f}")
            f1 = f1_score(targets, predictions, average='macro', zero_division=0)
            task_f1[i] = f1
            print(classification_report(targets, predictions, zero_division=0))
            
                        # Compute micro ROC AUC if more than 1 class
            if e - s > 1:
                try:
                    y_true = targets
                    y_score = all_predictions[:, s:e]
                    # One-hot encode targets for ROC AUC
                    y_true_oh = np.eye(e - s)[y_true.astype(int)]
                    roc_auc = roc_auc_score(y_true_oh, y_score, average='micro', multi_class='ovr')
                    task_roc_auc[i] = roc_auc
                    print(f"Task {i} - Micro ROC AUC: {roc_auc:.4f}")
                except Exception as ex:
                    print(f"Task {i} - ROC AUC calculation failed: {ex}")
            else:
                task_roc_auc[i] = np.nan

        num_epochs = 40
        # best_val_loss = float('inf')
        best_task_loss = task_losses.copy()
        print(f"Initial task losses: {best_task_loss}")
        # continue
        best_title = [None] * len(criterions)
        best_model = [None] * len(criterions)

        
        for epoch in range(num_epochs):
            total_loss = 0.0
            model.train()
            for batch_idx, (x_batch, mask_batch, target_batch) in enumerate(train_loader):
                x_batch, mask_batch, target_batch = x_batch.to(device), mask_batch.to(device), target_batch.to(device)
                optimizer.zero_grad()
                logits, loss, losses = model(x_batch, target=target_batch.long(), criterions=criterions)
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            avg_train_loss = total_loss / len(train_loader)
            print(f"[Train] Epoch {epoch+1}, Average Loss: {avg_train_loss:.4f}")
            
            total_loss = 0.0
            task_losses = np.zeros(len(criterions), dtype=np.float32)
            all_predictions = []
            all_targets = []
            model.eval()
            for _ in range(10):
                for batch_idx, (x_batch, mask_batch, target_batch) in enumerate(valid_loader):
                    with torch.no_grad():
                        x_batch, mask_batch, target_batch = x_batch.to(device), mask_batch.to(device), target_batch.to(device)
                        logits, loss, losses = model(x_batch, target=target_batch.long(), criterions=criterions)
                        total_loss += loss.item()
                        task_losses += losses
                        all_predictions.append(logits.cpu().numpy())
                        all_targets.append(target_batch.cpu().numpy())
            
            avg_val_loss = total_loss / len(valid_loader)
            
            print(f"[Validation] Epoch {epoch+1}, Average Loss: {avg_val_loss:.4f}")
            
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            task_f1 = np.zeros(len(criterions), dtype=np.float32)
            task_roc_auc = np.zeros(len(criterions), dtype=np.float32)
            for i, (s, e, _, _) in enumerate(criterions):
                predictions = np.argmax(all_predictions[:, s:e], axis=-1)
                targets = all_targets[:, i]
                # accuracy = accuracy_score(target, predictions)
                print(f"Task {i} - Loss: {task_losses[i]:.4f}")
                f1 = f1_score(targets, predictions, average='macro', zero_division=0)
                task_f1[i] = f1
                print(classification_report(targets, predictions, zero_division=0))
                
                            # Compute micro ROC AUC if more than 1 class
                if e - s > 1:
                    try:
                        y_true = targets
                        y_score = all_predictions[:, s:e]
                        # One-hot encode targets for ROC AUC
                        y_true_oh = np.eye(e - s)[y_true.astype(int)]
                        roc_auc = roc_auc_score(y_true_oh, y_score, average='micro', multi_class='ovr')
                        task_roc_auc[i] = roc_auc
                        print(f"Task {i} - Micro ROC AUC: {roc_auc:.4f}")
                    except Exception as ex:
                        print(f"Task {i} - ROC AUC calculation failed: {ex}")
                else:
                    task_roc_auc[i] = np.nan
                
            for i, (task_loss, f1, roc_auc) in enumerate(zip(task_losses, task_f1, task_roc_auc)):
                if task_loss < best_task_loss[i]:
                    best_task_loss[i] = task_loss
                    best_title[i] = f"epoch_{epoch+1}_loss_{task_loss:.4f}_f1_{f1:.4f}_roc_{roc_auc}_mode_{mode}"
                    best_model[i] = deepcopy(model.state_dict())
                    print(f"New best model found for task {i} at epoch {epoch+1} with loss: {task_loss:.4f}")

            
            if configs.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "task_losses": {f"task_{PREDICTING_FIELDS[i]}": loss for i, loss in enumerate(task_losses)},
                    "task_f1": {f"task_{PREDICTING_FIELDS[i]}": f1 for i, f1 in enumerate(task_f1)},
                })
                
        # Save the best model weights for each task
        for i, (title, model_weights) in enumerate(zip(best_title, best_model)):
            if model_weights is not None:
                torch.save(model_weights, f"weights/{PREDICTING_FIELDS[i]}/{title}.pth")
                print(f"Best model for task {i} saved as: {title}.pth")
                
        if configs.use_wandb:
            wandb.finish()

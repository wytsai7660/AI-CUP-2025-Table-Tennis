# Play years 執行說明

請先使用以下命令下載官方資料集：

```bash
bash download_dataset.sh
```

要復現 `play_years.csv` 中的結果，請使用以下命令：

```bash
bash make_prediction.sh
```

腳本會自動下載訓練完成的模型並進行預測。預測結果將輸出到 `play_years.csv` 檔案中。

如要自行訓練模型，請使用以下命令：

```bash
bash download_dataset.sh
python train.py
```
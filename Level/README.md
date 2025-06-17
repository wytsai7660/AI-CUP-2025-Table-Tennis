# Level 執行說明

請先使用以下命令下載官方資料集：

```bash
bash download_dataset.sh
```

要復現 `level.csv` 中的結果，請使用以下命令：

```bash
python3 test.py
```

預測結果將輸出到 `level.csv` 檔案中。

如要自行訓練模型，請使用以下命令下載資料集並進行訓練：

```bash
bash download_dataset.sh
python train.py
```
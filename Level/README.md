# Level 執行說明

要復現 `test_results_level_0.55751804.csv` 中的結果，請使用以下命令：

```bash
python3 test.py
```

預測結果將輸出到 `test_results.csv` 檔案中。

如要自行訓練模型，請使用以下命令下載資料集並進行訓練：

```bash
bash download_dataset.sh
python train.py
```
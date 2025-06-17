# Hold Racket Hand 執行說明

請先使用以下命令下載官方資料集：

```bash
bash download_dataset.sh
```

要訓練 Random Forest 模型，請使用以下命令：

```bash
python baseline.py
```

訓練完成後，使用 `python test.py` 進行預測並將結果輸出到 `hand.csv` 檔案中。

注意 `scikit-learn` 版本需要等於 1.5.2，否則可能無法準確復現。

可以直接下載已經訓練好的模型：

```bash
wget -q https://huggingface.co/alan314159/AI-CUP-2025-table-tennis/resolve/main/classifier.pkl
```
並使用以下命令進行預測：

```bash
python test.py
```

來取得可復現的預測結果。
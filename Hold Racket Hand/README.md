# Hold Racket Hand 執行說明

要訓練 Random Forest 模型，請使用以下命令：

```bash
bash make_prediction.sh
```

腳本會自動下載所需資料集並進行訓練。訓練完成後，將進行預測並將結果輸出到 `submission.csv` 檔案中。

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
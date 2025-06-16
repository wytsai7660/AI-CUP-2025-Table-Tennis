# AI CUP 2025 Table Tennis Player Information Prediction

本專案為 AI CUP 2025 桌球選手資訊預測競賽的程式碼，包含四個子任務的預測模型：
- 性別預測
- 持拍手預測
- 球齡預測
- 技能等級預測

## 環境設置

### 系統需求
- Python 3.12+
- Git
- wget

### 安裝依賴

使用 uv:
```bash
pip install uv
uv sync
```

主要依賴包括：
- torch>=2.7.1
- scikit-learn>=1.7.0 (Hold Racket Hand 需要 1.5.2 版本)
- pandas>=2.3.0
- numpy>=2.3.0
- matplotlib>=3.10.3
- 其他依賴請參考 `pyproject.toml`

## 復現結果

在專案根目錄執行以下命令，將自動運行所有子任務並合併結果：

```bash
./run_all_predictions.sh
```

關於各個子任務的詳細說明和訓練腳本執行方式，請參考各自目錄下的 `README.md` 文件。

## 檔案結構
```
AI-CUP-2025-Table-Tennis/
├── Gender/                 # 性別預測
├── Hold Racket Hand/       # 持拍手預測
├── Level/                  # 技能等級預測
├── Play_years/             # 球齡預測
├── pyproject.toml          # 專案依賴配置
├── merge_predictions.py    # 結果合併腳本
├── run_all_predictions.sh  # Linux 自動執行腳本
└── README.md              # 本說明文件
```

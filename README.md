# K-Nearest Neighbors (KNN)

[![Python](https://img.shields.io/badge/Python-3-3776AB)](https://www.python.org/)

## 作業描述

本次作業要實作 K-Nearest Neighbors Algorithm，可以用 Python 的 `csv` library 去讀寫 `.csv` 檔案，但**演算法的部分需手刻**，不可使用 `sklearn` 或其他現成的 library 建立。

本次作業分為以下步驟：

### 了解資料

下載資料集，理解題目的意義。
參閱上述的 column 定義，也可以分析與統計各個 column 的資料，思考如何處理。

### 前處理

由於有不同種的資料，在進行後續的模型預測之前，需要先將各種資料進行前處理，例如轉換成模型可以運算的型別。
這邊可以嘗試設計不同的前處理方式，來達到更高的準確度。

### 建立與訓練模型

按照課堂上所學，實作 K-Nearest Neighbors Algorithm，以 train split 的資料訓練，再對 validation 與 test split 進行預測。

### 評估與優化

利用 validation split 的資料進行預測與評分，想辦法提升準確度，例如調整模型的參數或是嘗試加上特殊的前處理方式。

## 實作結果

訓練集切割後的準確率與測試資料的準確率約為 70% 至 75%。

## 專案結構

```
knn/
├── knn.py          # KNN 演算法主程式
├── eval.py         # 評測準確率工具
├── data/           # 資料集
│   ├── train.csv   # 訓練集
│   ├── train_gt.csv# 訓練集標籤
│   ├── val.csv     # 測試集
│   └── val_gt.csv  # 測試集標籤
├── .gitignore
└── README.md
```

## 執行方式

```bash
# 執行 KNN 預測
python knn.py

# 評測準確率
python eval.py val_gt.csv val_pred.csv
```

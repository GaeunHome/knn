# K-Nearest Neighbors (KNN)

**課程名稱**：人工智慧導論  
**授課教授**：朱威達 教授

## Assignment Description

本次作業要實作 K-Nearest Neighbors Algorithm，可以用 Python 的 `csv` library 去讀寫 `.csv` 檔案，但 **演算法的部分需手刻**，不可使用 `sklearn` 或其他現成的 library 建立。

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

## 檔案用途

1. **eval.py**：評測經 KNN 預測之測試資料集與實際之測試資料集兩者準確率，指令為 `python eval.py val_gt.csv val_pred.csv`
2. **train.csv**：訓練集
3. **train_gt.csv**：訓練集之標籤
4. **val.csv**：測試集
5. **val_gt.csv**：測試集之標籤

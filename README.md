# knn
<p>課程名稱：人工智慧導論</p>
<p>授課教授：朱威達 教授</p>

# Assignment Description
本次作業要實作 K-Nearest Neighbors Algorithm，可以用 python 的 csv library 去讀寫 .csv 檔案，但<b>演算法的部分需手刻</b>，不可使用 sklearn 或其他現成的 library 建立。<br>
<br>
本次作業分為五個步驟：<br>
<h4>1. 了解資料</h4>
下載資料集，理解題目的意義。
參閱上述的 column 定義，也可以分析與統計各個 column 的資料，思考如何處理。
<h4>2. 前處理</h4>
由於有不同種的資料，在進行後續的模型預測之前，需要先將各種資料進行前處理，例如轉換成模型可以運算的型別。
這邊可以嘗試設計不同的前處理方式，來達到更高的準確度。
<h4>3. 建立與訓練模型</h4>
按照課堂上所學，實作 K-Nearest Neighbors Algorithm，以 train split 的資料訓練，再對 validation 與 test split 進行預測。
<h4>4. 評估與優化</h4>
利用 validation split 的資料進行預測與評分，想辦法提升準確度，例如調整模型的參數或是嘗試加上特殊的前處理方式。

# 實作結果
<p>訓練集切割後的準確率與測試資料的準確率約為70%至75%。</p>

# 檔案用途
<p>1. eval.py：評測經KNN預測之測試資料集與實際之測試資料集兩者準確率。</p>
<p>2. train.csv：訓練集</p>
<p>3. train_gt.csv：訓練集之標籤</p>
<p>4. val.csv：測試集</p>
<p>5. val_gt.csv：測試集之標籤</p>

import csv
import math
import random

def load_data(file_path):
    """
    讀取 CSV 檔案並返回資料列表。
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # 讀取標題行
        for row in reader:
            data.append(row)
    return headers, data

def merge_data(features, labels):
    """
    合併特徵資料和標籤資料。
    """
    merged_data = []
    for feature_row, label_row in zip(features, labels):
        merged_data.append(feature_row + label_row)
    return merged_data

def handle_missing_values(data):
    """
    處理資料中的缺失值。數值型欄位使用均值填補，類別型欄位使用眾數填補。
    如果某個欄位完全缺失，則填充為預設值。
    """
    num_rows = len(data)  # 取得資料的行數，假設 data 是一個二維列表（每行是一個資料紀錄）
    num_cols = len(data[0])  # 取得資料的欄位數，假設所有行的欄位數量是一致的

    # 遍歷每一個欄位 (從第 0 欄到第 num_cols - 1 欄)
    for j in range(num_cols):
        # 取得當前欄位的所有資料
        column_values = [row[j] for row in data]
        
        # 檢查欄位中是否所有值都是空字串或空格（即完全缺失）
        if all(value == '' or value == ' ' for value in column_values):
            # 如果該欄位完全缺失（所有值都是空字串或空格），進行處理
            for i in range(num_rows):
                if j in [4, 17, 18]:  # 假設欄位 4, 17, 18 是數值型欄位，代表 `tenure`, `MonthlyCharges`, `TotalCharges`
                    data[i][j] = '0.0'  # 如果是數值型欄位，填充為 '0.0'，這裡是假設使用字符串形式表示數值
                else:
                    data[i][j] = 'No'  # 否則，假設為類別型欄位，填充為 'No'
        else:
            # 如果欄位有非缺失值，則處理缺失值（空字串或空格）
            for i in range(num_rows):
                if data[i][j] == '' or data[i][j] == ' ':
                    if j in [4, 17, 18]:  # 處理數值型欄位
                        # 過濾掉缺失值，並將剩餘的數值轉換為浮點數型態
                        column_values = [float(row[j]) for row in data if row[j] not in ['', ' ']]
                        if column_values:  # 如果有有效的數值
                            # 計算數值型欄位的均值，並填補缺失值
                            data[i][j] = str(statistics.mean(column_values))
                    else:
                        # 處理類別型欄位
                        # 過濾掉缺失值，並取得剩餘的類別值
                        column_values = [row[j] for row in data if row[j] not in ['', ' ']]
                        if column_values:  # 如果有有效的類別值
                            # 計算類別型欄位的眾數，並填補缺失值
                            most_common_value = statistics.mode(column_values)
                            data[i][j] = most_common_value
                        else:
                            # 如果沒有有效的類別值，則填補為預設的 'No'
                            data[i][j] = 'No'
    # 返回處理過後的資料
    return data

# 將資料進行前處理，轉換成數值型別。
# 針對tenure、MonthlyCharges、TotalCharges做標準化
def preprocess_data(data, is_train=True):
    processed_data = []
    for row in data:
        processed_row = []
        # 標準化
        # Extract columns for normalization
        tenure_col = [int(row[4]) for row in data]
        monthly_charges_col = [float(row[17]) for row in data]
        total_charges_col = [float(row[18]) if row[18] != '' else 0.0 for row in data]

        # Calculate min and max for each column
        tenure_min, tenure_max = min(tenure_col), max(tenure_col)
        monthly_charges_min, monthly_charges_max = min(monthly_charges_col), max(monthly_charges_col)
        total_charges_min, total_charges_max = min(total_charges_col), max(total_charges_col)

        # gender: Male -> 0, Female -> 1
        processed_row.append(0 if row[0] == 'Male' else 1)
        # SeniorCitizen: already numeric
        processed_row.append(1 if row[1] == 'Yes' else 0)
        # Partner: Yes -> 1, No -> 0
        processed_row.append(1 if row[2] == 'Yes' else 0)
        # Dependents: Yes -> 1, No -> 0
        processed_row.append(1 if row[3] == 'Yes' else 0)
        # tenure
        tenure = int(row[4])
        tenure_normalized = (tenure - tenure_min) / (tenure_max - tenure_min)
        processed_row.append(tenure_normalized)
        # PhoneService: Yes -> 1, No -> 0
        processed_row.append(1 if row[5] == 'Yes' else 0)
        # MultipleLines: Yes -> 1, No -> 0, No phone service -> -1
        if row[6] == 'Yes':
            processed_row.append(1)
        elif row[6] == 'No':
            processed_row.append(0)
        else:
            processed_row.append(-1)
        # InternetService: Fiber optic -> 2, DSL -> 1, No -> 0
        if row[7] == 'Fiber optic':
            processed_row.append(2)
        elif row[7] == 'DSL':
            processed_row.append(1)
        else:
            processed_row.append(0)
        # OnlineSecurity, OnlineBackup, DeviceProtection, Techsupport, StreamingTV, StreamingMovies
        # Yes -> 1, No -> 0, No internet service -> -1
        for i in range(8, 14):
            if row[i] == 'Yes':
                processed_row.append(1)
            elif row[i] == 'No':
                processed_row.append(0)
            else:
                processed_row.append(-1)
        # Contract: Month-to-month -> 0, One year -> 1, Two year -> 2
        if row[14] == 'Month-to-month':
            processed_row.append(0)
        elif row[14] == 'One year':
            processed_row.append(1)
        else:
            processed_row.append(2)
        # PaperlessBilling: Yes -> 1, No -> 0
        processed_row.append(1 if row[15] == 'Yes' else 0)
        # PaymentMethod: map to integers
        payment_methods = {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }
        processed_row.append(payment_methods[row[16]])
        # MonthlyCharges: normalize
        monthly_charges = float(row[17])
        monthly_charges_normalized = (monthly_charges - monthly_charges_min) / (monthly_charges_max - monthly_charges_min)
        processed_row.append(monthly_charges_normalized)
        # TotalCharges: normalize, handle empty strings
        try:
            total_charges = float(row[18])
        except ValueError:
            total_charges = 0.0
        total_charges_normalized = (total_charges - total_charges_min) / (total_charges_max - total_charges_min)
        # Churn: Yes -> 1, No -> 0 (only for training data)
        if is_train:
            processed_row.append(1 if row[19] == 'Yes' else 0)
        
        processed_data.append(processed_row)
    return processed_data

def euclidean_distance(row1, row2, max_dist=float('inf')):
    """
    計算兩個資料點之間的歐幾里得距離。在距離超過max_dist時提前終止。
    """
    distance = 0.0
    for i in range(len(row1) - 1):  # 不包括標籤
        distance += (row1[i] - row2[i]) ** 2
        if distance > max_dist:     # 提前終止
            return float('inf')
    return math.sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    """
    找到最接近的鄰居。
    """
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    """
    預測分類。
    """
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def knn(train, test, num_neighbors):
    """
    使用 KNN 進行預測。
    """
    predictions = []
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions

def evaluate_accuracy(predictions, actual):
    """
    計算預測的準確度。
    """
    correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
    return correct / len(actual)

def save_processed_data(file_path, headers, data):
    """
    將前處理後的資料寫入新的 CSV 檔案。
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 寫入標題行
        writer.writerows(data)    # 寫入資料

def save_predictions(file_path, predictions):
    """
    將預測結果寫入 CSV 檔案。
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Churn'])  # 寫入標題行
        for prediction in predictions:
            # 將 1 轉換為 Yes， 0 轉換為 No
            writer.writerow(['Yes' if prediction == 1 else 'No'])

if __name__ == "__main__":
    # 設定訓練資料檔案路徑
    file_feature_path = 'train.csv'
    file_label_path = 'train_gt.csv'
    output_file_path = 'train_processed.csv'
    # 設定測試資料檔案路徑
    file_val_path = 'val.csv'
    file_test_path = 'test.csv'
    # 讀取資料
    feature_headers, feature_data_1 = load_data(file_feature_path) # 訓練資料
    label_headers, label_data = load_data(file_label_path)         # 訓練資料之結果
    val_headers, val_data_1 = load_data(file_val_path)
    test_headers, test_data_1 = load_data(file_test_path) 
    feature_data = handle_missing_values(feature_data_1)           # 訓練資料缺值處理
    val_data = handle_missing_values(val_data_1)                   # val.csv缺值處理
    test_data = handle_missing_values(test_data_1)                 # test.csv缺值處理
    # 合併訓練資料
    merged_data = merge_data(feature_data, label_data)
    # 前處理訓練、測試用資料、test.csv
    processed_train_data = preprocess_data(merged_data)
    processed_val_data = preprocess_data(val_data, is_train=False)
    processed_test_data = preprocess_data(test_data, is_train=False)
    # # 檢查前處理後的資料
    # for row in processed_train_data[:5]:  # 只顯示前5筆資料
    #     print(row)
    # # 檢查訓練處理後的資料
    # for row in processed_test_data[:5]:  # 只顯示前5筆資料
    #     print(row)
    # 儲存前處理資料 # 以檢查確認是否有誤
    save_processed_data(output_file_path, feature_headers + label_headers, processed_train_data)

    # 1 #
    # 分割資料為訓練集和驗證集
    # random.shuffle(processed_train_data)
    # split_ratio = 0.8
    # train_size = int(split_ratio * len(processed_train_data))
    # train_set = processed_train_data[:train_size]
    # val_set = processed_train_data[train_size:]

    # 使用驗證集進行預測
    # num_neighbors = 20 # k值
    # predictions = knn(train_set, val_set, num_neighbors)
    # actual = [row[-1] for row in val_set]
    # accuracy = sum(1 for i in range(len(actual)) if actual[i] == predictions[i]) / len(actual)
    # print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # 2 -> 使用驗證資料進行預測 #
    num_neighbors = 11
    predictions = knn(processed_train_data, processed_val_data, num_neighbors)
    output_file_path = 'val_pred.csv'
    save_predictions(output_file_path, predictions)

    # 3  -> 使test資料進行預測 #
    predictions = knn(processed_train_data, processed_test_data, num_neighbors)
    output_file_path = 'test_pred.csv'
    save_predictions(output_file_path, predictions)
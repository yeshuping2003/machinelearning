# bike_sharing.py
# 随机森林回归器
# 计算2万多个观测值大约需要5分钟，显现出数据量对运算时间的影响了。
import sys
import csv

import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from housing import plot_feature_importances

def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'r'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[1:8]) # 数据0列是序号，9列是目标值房价，取1~8列。取少了过拟合，误差0，得分1。
        y.append(row[-1])   # 取列需要根据数据集的情况决定特征值 X 的列数。

    # Extract feature names
    feature_names = np.array(X[0]) # 特征的名称在第0行，分析结果时用。

    # Remove the first row because they are feature names
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

if __name__=='__main__':
    # Load the dataset from the input file
    """
    from sklearn.datasets.california_housing import fetch_california_housing
    housing=fetch_california_housing()
    X = housing.data
    y = housing.target
     该数据包含9个变量的20640个观测值，
     该数据集包含平均房屋价值作为目标变量和以下输入变量（特征）：
     平均收入、房屋平均年龄、平均房间、平均卧室、人口、平均占用、纬度和经度。
     ['MedInc',  'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
      'AveOccup', 'Latitude', 'Longitude', 'target']
    """    
    X, y, feature_names = load_dataset("housing_t.csv") # ("bike_day.csv") # (sys.argv[1])
    # 程序可以分析其它数据集，
    X, y = shuffle(X, y, random_state=7) 

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit Random Forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(X_train, y_train)

    # Evaluate performance of Random Forest regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    print ("\n#### Random Forest regressor performance ####")
    print ("Mean squared error =", round(mse, 2))
    print ("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)

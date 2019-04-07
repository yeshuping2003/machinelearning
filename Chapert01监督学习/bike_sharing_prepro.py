# 我的数据预处理程序
# 做成适合随机森林回归算法所需的数据集：
# 具有列名，前部列是特征列，最后1列是目标值，用逗号分隔的.csv文件。
# 供bike_sharing.py使用。
# 读取导入的数据集，合并data和 target ,
# 加上数据集,命名为 housing里的特征名 housing.feaure_names，
# 保存为 .csv文件。
# 验证的数据：housing.csv(california_housing), bike_day.csv, boston_housing_prices.csv 
# 读取数据集
from sklearn.datasets.california_housing import fetch_california_housing
housing=fetch_california_housing()
housing # 数据集描述
X = housing.data
y = housing.target
X.shape
y.shape
# 读取特征名
names = housing.feature_names
# 追加目标特征名
names.append('target')

# 合并DataFrame
import pandas as pd
X = pd.DataFrame(housing.data)
y = pd.DataFrame(housing.target)

# 合并 X和 y成为 df
df = pd.concat([X, y], axis=1)
# 保存文件
df.to_csv(filename, header=names)

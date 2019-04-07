# regressor.py
# 1.7 岭回归器

import numpy as np
import matplotlib.pyplot as plt
"""
import utilities
# Load input data
input_file = "data_multivar01_X4.txt"
X, y = utilities.load_data(input_file)
"""
filename = "data_singlevar.txt"#sys.argv[1]#"data_multivar01_X4.txt"#
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

#2.为了检验机器学习模型是否达到满意度，将数据分为训练和测试2组
# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

#3.已经准备好了训练模型，创建回归器对象
# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()

#用训练数据集训练线形回归器，向fit方法输入数据即可
# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

#4.检查拟合
y_train_pred = linear_regressor.predict(X_train)

# Plot outputs
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()


# Predict the output
y_test_pred = linear_regressor.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

# Measure performance
import sklearn.metrics as sm

ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print ( "Mean absolute error 平均误差=", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
print ( "Mean squared error 均方误差=", round(sm.mean_squared_error(y_test, y_test_pred), 2) )
print ( "Median absolute error 中位数绝对误差=", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print ( "Explain variance score 解释方差分best 1=", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print ( "R2 score R方得分，相关系数best 1=", round(sm.r2_score(y_test, y_test_pred), 2))

# Polynomial regression
# 1.8 多项式回归
# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [0.39,2.78,7.11]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print ("\nLinear regression:\n", linear_regressor.predict(datapoint))
print ("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

# Stochastic Gradient Descent regressor
sgd_regressor = linear_model.SGDRegressor(loss='huber', n_iter=50)
sgd_regressor.fit(X_train, y_train)
print ("\nSGD regressor:\n", sgd_regressor.predict(datapoint))

"""
以上是原文
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=3)
#3次多项式
X_train_transformed = polynomial.fit_transform(X_train)
#数据第一点测试

datapoint = [0.39,2.78,7.11]
poly_datapoint = polynomial.fit_transform([datapoint])

#例 input_data_encoded = input_data_encoded.reshape(1, len(input_data))
#poly_datapoint = poly_datapoint.reshape(1,len(poly_datapoint))

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

print ("\nLinear regression:\n", linear_regressor.predict((datapoint)[0]))

print(poly_datapoint.shape)
print(poly_datapoint)
poly_datapoint = poly_datapoint.reshape(1,len([poly_datapoint]))
print ("\nPolynomial regression:\n", poly_linear_model.predict((poly_datapoint)[0]))
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
#读取数据
train=pd.read_csv('train.csv')
features=['GrLivArea','OverallQual','GarageCars']
X=train[features].values
y=train['SalePrice'].values
#对不同特征做归一化处理

scaler=StandardScaler()
X=scaler.fit_transform(X)

y=(y-y.mean())/y.std()
X=np.c_[np.ones(X.shape[0]),X]
n=len(y)
#代价函数
def loss(X,y,theta):
    return 1/(2*n)*np.sum((X.dot(theta)-y)**2)
#记录损失的历史
#梯度更新
def gradient_descent(X,y,theta,lr=0.01,epoches=100):
    co_history = []
    for i in range(epoches):
        predictions=X.dot(theta)
        errors=predictions-y
        gradient=X.T.dot(errors)/n
        theta=theta-lr*gradient
        cost=loss(X,y,theta)
        co_history.append(cost)
    return theta,co_history
#训练过程
theta=np.zeros(X.shape[1])
lr=0.01
epoches=5000
theta,co_history=gradient_descent(X,y,theta,lr,epoches)

#可视化
plt.plot(range(len(co_history)),co_history)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('cost')
plt.show()

x_test=X[0]
y_test=X[0].dot(theta)
print("预测值（标准化后）:", y_test)
print("真实值（标准化后）:", y[0])
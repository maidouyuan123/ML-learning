import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


train=pd.read_csv('train.csv')

features=['Pclass','Sex','Age','Fare']

#统计缺失值的数量
#print(train[features].isnull().sum())
train.fillna({
    'Age': train['Age'].median(),
    #'Cabin': 'Unknown'
}, inplace=True)

#print(train[features].isnull().sum())
#把非数值做数值化处理
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
x=train[features].values
y=train['Survived'].values
#特征的归一化处理
scaler=StandardScaler()
x=scaler.fit_transform(x)
x=np.c_[np.ones(x.shape[0]),x]
n=len(y)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(x,y,theta):
    return -np.sum(y*np.log(sigmoid(x.dot(theta)))-(1-y)*np.log(1-sigmoid(x.dot(theta))))
def gradient(x,y,theta):
    #return x.T.dot(x.dot(theta)-y) 这是错误代码
    m = len(y)
    h = sigmoid(x.dot(theta))
    return (1 / m) * x.T.dot(h - y)
def gradient_descent(x,y,theta,lr=0.01,epoches=100):
    m=len(y)
    co_history=[]
    for i in range(epoches):
        predictions = x.dot(theta)
        predictions = sigmoid(predictions)
        errors = predictions - y
        gradient = x.T.dot(errors) / m
        theta = theta - lr * gradient
        cost = loss(x, y, theta)
        co_history.append(cost)
    return theta, co_history

theta=np.zeros(x.shape[1])
lr=0.01
epoches=50000
theta,co_history=gradient_descent(x,y,theta,lr,epoches)

#可视化
plt.plot(range(len(co_history)),co_history)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('cost')
plt.show()

p = sigmoid(x @ theta)
y_pred = (p >= 0.5).astype(int)
acc = (y_pred == y).mean()
print("Train accuracy:", acc)
print("Final loss:", co_history[-1])
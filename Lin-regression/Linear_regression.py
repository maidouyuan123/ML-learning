import numpy as np
import matplotlib.pyplot as plt

def loss(y_hat,y):
    return np.sum((y_hat-y)**2)/y.shape
def gradient(x,y,w,b):#通常情况下要使用loss函数，但是这里我们因为函数简单直接推导出来了导数。
    dw=np.sum((x*w+b-y)*x)/y.shape[0]
    db=np.sum(x*w+b-y)/y.shape[0]
    return dw,db
def gradient_descent(grad,w,b,lr=0.01):
    w-=lr*grad[0]
    b-=lr*grad[1]
    return w,b

true_w,true_b,epoch=input("请输入3个数，分别代表真实的直线参数w和b以及训练次数epoch：").split()
true_w=int(true_w)
true_b=int(true_b)
epoch=int(epoch)
x=np.random.normal(0,1,1000)#生成一千个数据
y=x*true_w+true_b
y=y+np.random.random(len(x))

w=0
b=0
for i in range(epoch):
    grad=gradient(x,y,w,b)
    w,b=gradient_descent(grad,w,b)
plt.scatter(x,y,color="blue",label="data points")
plt.plot(x,w*x+b,color="green",label="linear regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
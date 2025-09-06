import numpy as np
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#数学公式算梯度
def lr_cost_grad(theta,x,y,lam):
    m=x.shape[0]
    eps=1e-12
    h=sigmoid(np.dot(x,theta))
    cost=-1/m*(np.dot(y,np.log(h+eps))+np.dot(1-y,np.log(1-h+eps)))
    grad=np.dot(x.T,h-y)/m
    grad[1:]+=lam/m*theta[1:]
    cost+=lam/(2*m)*np.sum(theta[1:]**2)
    return cost,grad
#更新对于k的判断的梯度
def gd(theta,x,y,lam=0.1,lr=0.5,iters=600):
    for _ in range(iters):
        c,g=lr_cost_grad(theta,x,y,lam)
        theta=theta-lr*g
    return theta
#分别判断是不是1、2、3......
def one_vs_all(k,x,y,lam=0.1,lr=0.5,iters=600):
    m,n=x.shape
    X=np.c_[np.ones((m,1)),x]
    all_theta=np.zeros((k,n+1))
    for i in range(k):
        yk=(y==i).astype(float)
        theta=np.zeros(n+1)
        all_theta[i]=gd(theta,X,yk,lam,lr,iters)
    return all_theta
#预测
def predice_one_vs_all(all_theta,x):
    m, n = x.shape
    X = np.c_[np.ones((m, 1)), x]
    probs=sigmoid(np.dot(X,all_theta.T))
    return np.argmax(probs,axis=1)
train=pd.read_csv("mnist_train.csv")

x_train=train.iloc[:,:-1].values/255.0
y_train=train.iloc[:,0].values
x_train = x_train[:1000]
y_train = y_train[:1000]
test = pd.read_csv("mnist_test.csv")
X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values
X_test = X_test[:100]
y_test = y_test[:100]

k=10
all_theta=one_vs_all(k,x_train,y_train)
y_pred = predice_one_vs_all(all_theta, X_test)
acc = (y_pred == y_test).mean()
print("Test accuracy:", acc)
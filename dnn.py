import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
from testCases_v2 import *
from planar_utils import plot_decision_boundary,sigmoid,load_extra_datasets,load_planar_dataset

np.random.seed(1)

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # 样本数量
    N = int(m/2) # 每个类别的样本量
    D = 2 # 维度数
    X = np.zeros((m,D)) # 初始化X
    Y = np.zeros((m,1), dtype='uint8') # 初始化Y
    a = 4 # 花儿的最大长度

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


class deepNN():
    def __init__(self):
        self.l = 2
        self.ld = [2,4,1]
        m = self.get_x(x)[1]
        self.w = []
        self.b = []
        self.a = []
        self.z = []
        self.dw = []
        self.db = []
        self.da = []
        self.dz = []
        for i in range(self.l):
            self.a.append(np.random.randn(self.ld[i+1],m)*0.01)
            self.z.append(np.random.randn(self.ld[i+1],m)*0.01)
            self.w.append(np.random.randn(self.ld[i+1],self.ld[i])*0.01)
            self.b.append(np.random.randn(self.ld[i+1],1)*0.01)

            self.da.append(np.random.randn(self.ld[i+1],m)*0.01)
            self.dz.append(np.random.randn(self.ld[i+1],m)*0.01)
            self.dw.append(np.random.randn(self.ld[i+1],self.ld[i])*0.01)
            self.db.append(np.random.randn(self.ld[i+1],1)*0.01)

        #print(str(self.w)+"===========/n"+str(self.b)+ "init finished")

    def get_para(self, para):
        return para

    def get_x(self, x):
        return np.shape(x)

    def forPp(self,x): # i= 0..l-1 a_former 0..l-1 a[0] =
        self.z[0] = np.dot(self.w[0],x) + self.b[0]
        self.a[0] = np.tanh(self.z[0]) # 1..l a[l] = y
        self.z[1] = np.dot(self.w[1],self.a[0])+self.b[1] # z_later  1..l  a[0] = x
        self.a[1] = 1.0/(1.0+np.exp(-self.z[1])) # 1..l a[l] = y

    def backPp(self,x,y):
        m = self.get_x(x)[1]
        self.dz[1] = self.a[1] - y  # 确定最后一层是sigmoid
        self.dw[1] = 1.0 / m * np.dot(self.dz[1], self.a[0].T)
        self.db[1] = 1.0 / m * np.sum(self.dz[1], axis=1, keepdims=True)

        self.da[0] = np.dot(self.w[1].T, self.dz[1])  # 1 0
        gz = 1.0-np.power(self.a[0],2)
        self.dz[0] = self.da[0]*gz
        self.dw[0] = 1.0/m*np.dot(self.dz[0],x.T)
        self.db[0] = 1.0/m*np.sum(self.dz[0],axis=1,keepdims=True)

    def update(self,lr):
        for i in range(2):
            self.w[i] = self.w[i]-lr*self.dw[i]
            self.b[i] = self.b[i]-lr*self.db[i]

    def train(self,x,y,epoch,lr):
        for k in range(epoch):
            self.forPp(x)
            self.backPp(x,y)
            self.update(lr)

    def predict(self,x): # 0,1
        self.forPp(x)
        return np.where(self.a[1]>0.5,1,0)


x,y = load_planar_dataset()
y = y.ravel()
para = [4,1]
dnn = deepNN()
dnn.train(x,y,10000,3)
yhat = dnn.predict(x)
print(yhat.shape)

print("precision of nn:"+str(100-(np.sum(np.abs(yhat-y)))/4)+"%")

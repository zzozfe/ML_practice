import numpy as np
import pandas as pd
import sys

trainxcsv = sys.argv[3]
trainycsv = sys.argv[4]
testcsv = sys.argv[5]
anscsv = sys.argv[6]

train_X = pd.read_csv(trainxcsv)
train_Y = pd.read_csv(trainycsv,header = None)
test_X = pd.read_csv(testcsv)
train_X['fnlwgt'] = train_X['fnlwgt'].clip(0,800000)
test_X['fnlwgt'] = test_X['fnlwgt'].clip(0,800000)

def stander(x):
    for c in x.columns:
        mean = x[c].mean()
        std = x[c].std()
        if std != 0 :
            x[c] = x[c].map(lambda x : (x-mean)/std)
    return x    

def pro(x):
    if x > 0 :
        x = 1
    else:
        x = 0
    return x    

def lse(x,y):
    x = x.to_numpy()
    y = y.to_numpy()
    w = np.full(106,0.1).reshape(-1,1)
    b = 0.1
    lr = 0.001
    lamda = 0.001
    itea = 10000
    sigma = np.zeros((106,1))
    sigmab = 0
    v = np.zeros((106,1))
    vb = 0
    beta1 = 0.9
    beta2 = 0.999
    eplision = 1e-8
    t = 0

    for i in range(itea):
        t += 1
        z = np.dot(x, w) + b
        y_hat = 1/(1+np.exp(-z))
        L = y - y_hat
        w_grad = -1*np.dot(x.T,L)
        b_grad = -1*L.sum()
        sigma = beta2*sigma + (1-beta2)*w_grad*w_grad
        sigmab = beta2*sigmab + (1-beta2)*b_grad*b_grad
        v = beta1*v + (1-beta1)*w_grad
        vb = beta1*vb + (1-beta1)*b_grad
        sigman = sigma/(1-beta2**t)
        sigmabn = sigmab/(1-beta2**t)
        vn = v/(1-beta1**t)
        vbn = vb/(1-beta1**t)
    
        b = b - (lr*vbn)/(np.sqrt(sigmabn) + eplision)
        w = w - (lr*vn)/(np.sqrt(sigman)+ eplision)

    return w,b

conx = pd.concat((train_X, test_X))  
conx['capital_gain'] = conx['capital_gain'].map(pro)
conx['capital_loss'] = conx['capital_loss'].map(pro)
conx = stander(conx)
train_X = conx.iloc[0:train_X.shape[0],:]
test_X = conx.iloc[train_X.shape[0]::,:]

w,b = lse(train_X,train_Y)

predict = pd.DataFrame()
ids = []
values = []
z = np.dot(test_X, w) + b
pc0x = 1/(1 + np.exp(-z))
for i in range(len(pc0x)):
    ids.append(i+1)
    if pc0x[i] > 0.5:
        values.append(1)
    else:
        values.append(0)
predict['id'] = ids
predict['label'] = values

predict.to_csv(anscsv,index=False)
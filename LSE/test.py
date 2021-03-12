import pandas as pd
import numpy as np
import sys

testcsv = sys.argv[1]
anscsv = sys.argv[2]

w = pd.read_csv('w.csv')
w = w.to_numpy()
b = w[-1]
w = w[0:-1]
test = pd.read_csv(f'{testcsv}')

for i in range(9):
    test.loc[:,str(i)] = test.loc[:,str(i)].str.replace('#','')
    test.loc[:,str(i)] = test.loc[:,str(i)].str.replace('x','')
    test.loc[:,str(i)] = test.loc[:,str(i)].str.replace('*','')
test = test.replace({'NR':0})
columns = []
for i in range(9):
    columns.append(str(i))
test[columns] = test[columns].astype(np.float)

test = test.fillna(0)
test = test.drop(['測項','id'],axis=1)
test = test.to_numpy()
test = test.reshape(-1,162)

predict = pd.DataFrame(columns = ['id','value'])
ids = []
values = []
for i in range(500):
    ids.append('id_'+ str(i))
    values.append(float(np.dot(test,w)[i]+b))
    
predict['id'] = ids
predict['value'] = values
predict.to_csv(f'{anscsv}' , index = False)
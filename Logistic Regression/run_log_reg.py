import pandas as pd
import numpy as np
from test import Logistic_Regression
import sys

train_data = pd.read_csv('bank-note/train.csv', header=None)
# process data
raw = train_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
train_x = np.copy(raw)
train_x[:,num_col - 1] = 1
train_y = raw[:, num_col - 1]
train_y = 2 * train_y - 1

print(train_x)
print(train_y)

test_data = pd.read_csv('bank-note/test.csv', header=None)
raw = test_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
test_x = np.copy(raw)
test_x[:,num_col - 1] = 1
test_y = raw[:, num_col - 1]
test_y = 2 * test_y - 1

gamma_set = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
model= Logistic_Regression()
v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

for v in v_list:
    model.set_v(v)
    print('Variane:', v)
    if sys.argv[1] == "map": w= model.train_MAP(train_x, train_y)
    elif sys.argv[1] == "mle": w= model.train_ML(train_x, train_y)


    pred = np.matmul(train_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('train_error: ', train_err, ' test_error: ', test_err)
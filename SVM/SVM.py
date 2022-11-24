import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from prim_SVM import PrimSVM
from dual_SVM import DualSVM


bank_train_df = pd.read_csv('bank-note/train.csv', header=None)
bank_test_df = pd.read_csv('bank-note/test.csv', header=None)

bank_train_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

X_train = bank_train_df.iloc[:, :-1].values
Y_train = bank_train_df.iloc[:, -1].values

Y_train[Y_train == 0] = -1

X_test = bank_test_df.iloc[:, :-1].values
Y_test = bank_test_df.iloc[:, -1].values

Y_test[Y_test == 0] = -1


lr = 0.001
a = 0.001
T = 100
list_C = [100/873, 500/873, 700/873]
list_gamma = [0.01, 0.1, 0.5, 1, 5, 100]
weights = []

prim_svm_para = defaultdict(list)
prim_svm_train_test_err = defaultdict(list)


print("Primal SVM :")
print('Changing learning rate with a :')
for C in tqdm(list_C):
    prim_a = PrimSVM("lr_a", a, lr=lr, C=C)
    prim_a.fit(X_train, Y_train, T)
    pred_train = prim_a.predict(X_train)
    pred_test = prim_a.predict(X_test)

    print('C: ' + str(C))
    print('Training Error: ' + str(prim_a.evaluate(X_train, Y_train)))
    print('Testing Error: ' + str(prim_a.evaluate(X_test, Y_test)))
    prim_svm_train_test_err['a'].append({'train': prim_a.evaluate(X_train, Y_train), 'test': prim_a.evaluate(X_test, Y_test)})
    weights.append(prim_a.w)
    prim_svm_para["a"].append(prim_a.w)
    print()

print(prim_svm_para)
print(prim_svm_train_test_err)


print('Chaning learning rate with epoch :')
for C in tqdm(list_C):
    prim_epoch = PrimSVM("lr_epoch", a, lr=lr, C=C)
    prim_epoch.fit(X_train, Y_train)
    pred_train = prim_epoch.predict(X_train)
    pred_test = prim_epoch.predict(X_test)

    print('C: ' + str(C))
    print('Training Error: ' + str(prim_epoch.evaluate(X_train, Y_train)))
    print('Testing Error: ' + str(prim_epoch.evaluate(X_test, Y_test)))
    prim_svm_train_test_err['epoch'].append({'train': prim_epoch.evaluate(X_train, Y_train), 'test': prim_epoch.evaluate(X_test, Y_test)})
    weights.append(prim_epoch.w)
    prim_svm_para["epoch"].append(prim_epoch.w)
    print()


print('Differences between the weights learned from the two learning rate schedules :')
for i in range(len(prim_svm_para["a"])):
    print('C: ' + str(list_C[i]))
    print('Difference of weights: ' + str(np.linalg.norm(prim_svm_para["a"][i] - prim_svm_para["epoch"][i])))
    print()
print('\n')

# difference between the two methods
print('Differences between the train and test errors learned from the two learning rate schedules :')
for i in range(len(prim_svm_train_test_err["a"])):
    print('C: ' + str(list_C[i]))
    print('Difference of training error: ' + str(prim_svm_train_test_err["a"][i]['train'] - prim_svm_train_test_err["epoch"][i]['train']))
    print('Difference of testing error: ' + str(prim_svm_train_test_err["a"][i]['test'] - prim_svm_train_test_err["epoch"][i]['test']))
    print()
print('\n')

dual_svm_para = defaultdict(list)
print("Dual SVM :")
for C in tqdm(list_C):
    dual = DualSVM("linear", C=C)
    dual.fit(X_train, Y_train)
    pred_train = dual.predict(X_train)
    pred_test = dual.predict(X_test)
    print('C: ' + str(C))
    print('Training Error: ' + str(dual.evaluate(X_train, Y_train)))
    print('Testing Error: ' + str(dual.evaluate(X_test, Y_test)))
    weights.append(np.append(dual.w, dual.b))
    dual_svm_para["linear"].append(np.append(dual.w, dual.b))
    print("No. of support vectors: " + str(len(dual.support_vectors)))
    print()



print('Differences between the weights learned from the two learning rate schedules :')
for i in range(len(prim_svm_para["a"])):
    print('C: ' + str(list_C[i]))
    print('Difference of weights (a-linear): ' + str(np.linalg.norm(prim_svm_para["a"][i] - dual_svm_para["linear"][i])))
    print("Difference of weights (epoch-linear): " + str(np.linalg.norm(prim_svm_para["epoch"][i] - dual_svm_para["linear"][i])))
print()

C_500_873 = {}

for gamma in tqdm(list_gamma):
    for C in list_C:
        dual = DualSVM("gaussian", C=C, gamma=gamma)
        dual.fit(X_train, Y_train)
        pred_train = dual.predict(X_train)
        pred_test = dual.predict(X_test)
        print('C: ' + str(C) + ', gamma: ' + str(gamma))
        print('Training Error: ' + str(dual.evaluate(X_train, Y_train)))
        print('Testing Error: ' + str(dual.evaluate(X_test, Y_test)))
        weights.append(np.append(dual.w, dual.b))
        print("Weights and bias: ", dual.w, dual.b)

        print("No. of support vectors: " + str(len(dual.support_vectors)))

        if C == 500 / 873:
            C_500_873[gamma] = dual.support_vectors

        print()


print("No. of same support vectors (C = 500/873 and gamma = 0.1, and 0.5): " + str(
    len(set(C_500_873[0.1]).intersection(set(C_500_873[0.5])))))


print("No. of same support vectors (C = 500/873 and gamma = 0.1, and 0.01): " + str(
    len(set(C_500_873[0.1]).intersection(set(C_500_873[0.01])))))
import pandas as pd

from standard_per import *
from voted_per import *
from avg_perception import *


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

std_weights = std_per_train(X_train, Y_train, epochs=10, lr=0.1)
avg_std_percetron_error_rate = std_per_evaluate(X_test, Y_test, std_weights)

print("############ Standard Perceptron ############")
print("Standard perceptron: Learned weight vector: ", std_weights)
print("Average standard perceptron prediction error on the test dataset: ", avg_std_percetron_error_rate)
print("\n\n")

voted_weight_list, count_list = voted_per_train(X_train, Y_train, epochs=10, lr=0.1)
voted_weight_array = np.array(voted_weight_list)
voted_count_array = np.array(count_list)
avg_voted_percetron_error_rate = voted_per_evaluate(X_test, Y_test, voted_weight_array[1:], voted_count_array[1:])

print("############ Voted Perceptron ############")
with open("voted_weights_counts.txt", "w") as external_file:
    for i in range(1, len(voted_weight_list)):
        print("Distinct weight vector-",i," : ", voted_weight_list[i], file=external_file)
        print("Correctly predicted examples using weight vector-",i," : ", count_list[i], file=external_file)
        print("Distinct weight vector-",i," : ", voted_weight_list[i])
        print("Correctly predicted examples using weight vector-",i," : ", count_list[i])
external_file.close()
print("\n")
print("Voted learned weight vector: ", voted_weight_array[-1])
print("Average voted perceptron prediction error on the test dataset: ", avg_voted_percetron_error_rate)
print("\n\n")


a_weights, avg_weights = avg_per_train(X_train, Y_train, epochs=10, lr=0.1)
avg_percetron_error_rate = avg_per_evaluate(X_test, Y_test, a_weights)
print("############ Average Perceptron ############")
print("Learned weight vector: ", avg_weights)
print("Average prediction error on the test dataset: ", avg_percetron_error_rate)
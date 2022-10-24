import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from batch_gradient_descent_algo import Batch_Gradient_Descent
from stochastic_gradient_descent_algo import Stochastic_Gradient_Descent
from analytical_solution import Analytical_Solution

train_data = "concrete/train.csv"
test_data = "concrete/test.csv"

features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'output']

train_df = pd.read_csv(train_data, names=features).astype(float)
test_df = pd.read_csv(test_data, names=features).astype(float)

X_train = train_df.drop('output', axis=1)
y_train = train_df['output']

X_test = test_df.drop('output', axis=1)
y_test = test_df['output']

batch_gradient_descent = Batch_Gradient_Descent()

stochastic_gradient_descent = Stochastic_Gradient_Descent()

bgd_training_cost = batch_gradient_descent.optimize(X_train, y_train)
batch_gradient_descent_error = batch_gradient_descent.bgd_loss_func(X_test, y_test)
print('Batch Gradient Descent weights: ' + str(batch_gradient_descent.weight_vec))

sgd_training_cost = stochastic_gradient_descent.optimize(X_train, y_train)
stochastic_gradient_descent_error = stochastic_gradient_descent.sgd_loss_func(X_test, y_test)
print('Stochastic Gradient Descent weights: ' + str(stochastic_gradient_descent.weight_vec))

#Plot BGD cost and save in png
fig_bgd = plt.figure(1)
fig_bgd.suptitle('Gradient Descent Cost Rate')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.plot(bgd_training_cost, 'r')
plt.legend(['train'])
fig_bgd.savefig('Batch Gradient Descent Cost Rate.png')

#Plot SGD cost and save in png
fig_sgd = plt.figure(2)
fig_sgd.suptitle('Stochastic Gradient Descent Cost Rate')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.plot(sgd_training_cost, 'r')
plt.legend(['train'])
fig_sgd.savefig('Stochastic Gradient Descent Cost Rate.png')

#analytical Gradient Descent Solution
analytical_gradient_descent = Analytical_Solution(X_train, y_train)
analytical_gradient_descent_error = analytical_gradient_descent.analytical_loss_func(X_test, y_test)
print('Analytical Gradient Descent weight: ' + str(analytical_gradient_descent.weight_vec))
print('Analytical Gradient Descent Error: ' + str(analytical_gradient_descent_error))

#BGD and SGD weight errors
print('Batch Gradient Descent Weight Error: ' + str(np.linalg.norm(batch_gradient_descent.weight_vec - analytical_gradient_descent.weight_vec)))
print('Stochastic Gradient Descent Weight Error: ' + str(np.linalg.norm(stochastic_gradient_descent.weight_vec - analytical_gradient_descent.weight_vec)))

plt.show()
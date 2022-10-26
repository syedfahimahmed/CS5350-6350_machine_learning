from ada_boost import Ada_Boost
import pandas as pd
import matplotlib.pyplot as plt

from Decision_Tree.preprocessing import preprocessing_bank_data

train_data = "bank/train.csv"
test_data = "bank/test.csv"

bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

# bank data numerical columns
bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_df = pd.read_csv(train_data, names=bank_column_names)
test_df = pd.read_csv(test_data, names=bank_column_names)

train_df['y'] = train_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
test_df['y'] = test_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

# median of numerical attributes of bank train data
train_numerical_thresholds = train_df[bank_numerical_columns].median()
test_numerical_thresholds = test_df[bank_numerical_columns].median()

#  consider unknown as category
preprocessed_bank_train_df = preprocessing_bank_data(train_df, train_numerical_thresholds,
                                                            bank_numerical_columns)
preprocessed_bank_test_df = preprocessing_bank_data(test_df, test_numerical_thresholds, bank_numerical_columns)

print("Bank Dataset Evaluation (with unknown considered as value):")
Iteration = 500

ada = Ada_Boost(preprocessed_bank_train_df, preprocessed_bank_test_df, list(preprocessed_bank_train_df.columns[:-1]),
               preprocessed_bank_train_df['y'], 1, Iteration)

fig_tree = plt.figure(1)
fig_tree.suptitle('Tree Prediction Error')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Error Rate', fontsize=16)
plt.plot(ada.train_error_decision_tree, 'b', label='Training Error')
plt.plot(ada.test_error_decision_tree, 'r', label='Testing Error')
plt.legend(['train error', 'test error'], loc='upper right')
fig_tree.savefig('Tree Prediction Error.png')

fig_ada = plt.figure(2)
fig_ada.suptitle('Ada_Boost Error')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Error Rate', fontsize=16)
plt.plot(ada.train_error, 'b', label='Training Error')
plt.plot(ada.test_error, 'r', label='Testing Error')
plt.legend(['train error', 'test error'], loc='upper right')
fig_ada.savefig('Ada_Boost Error.png')
from bagged_tree_algo import Bagged_Tree
import pandas as pd
import matplotlib.pyplot as plt

from Decision_Tree.preprocessing import preprocessing_bank_data

training_data = "bank/train.csv"
testing_data = "bank/test.csv"

bank_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_data_frame = pd.read_csv(training_data, names=bank_columns)
test_data_frame = pd.read_csv(testing_data, names=bank_columns)

train_data_frame['y'] = train_data_frame['y'].apply(lambda x: '1' if x == 'yes' else '-1')
train_data_frame['y'] = train_data_frame['y'].astype(float)

test_data_frame['y'] = test_data_frame['y'].apply(lambda x: '1' if x == 'yes' else '-1')
test_data_frame['y'] = test_data_frame['y'].astype(float)


train_numerical_thresholds = train_data_frame[bank_numerical_columns].median()
test_numerical_thresholds = test_data_frame[bank_numerical_columns].median()


preprocessed_bank_train_df = preprocessing_bank_data(train_data_frame, train_numerical_thresholds,bank_numerical_columns)
preprocessed_bank_test_df = preprocessing_bank_data(test_data_frame, test_numerical_thresholds, bank_numerical_columns)

print("Bagged Tree Performance for Bank Dataset:")
Iteration = 500

bgt = Bagged_Tree(preprocessed_bank_train_df, preprocessed_bank_test_df, list(preprocessed_bank_train_df.columns[:-1]),
                 preprocessed_bank_train_df['y'], 100, Iteration)

range_of_trees = range(1, 501)

fig_bgt = plt.figure()
fig_bgt.suptitle('Bagged Decision Tree')
plt.xlabel('Number of trees', fontsize=16)
plt.ylabel('Error Rate', fontsize=16)
plt.plot(bgt.training_error, 'b', label='Training Error')
plt.plot(bgt.testing_error, 'r', label='Testing Error')
plt.legend()
fig_bgt.savefig('Bagged Decision Tree Error.png')
from random_forest_algo import RandomForest
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

print("Random Forest Performance for Bank Dataset:")
T = 500
x = range(1, T + 1)

rf_2 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                    list(preprocessed_bank_train_df.columns[:-1]),
                    preprocessed_bank_train_df['y'], 16, T, 2)

rf_4 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                    list(preprocessed_bank_train_df.columns[:-1]),
                    preprocessed_bank_train_df['y'], 16, T, 4)

rf_6 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                    list(preprocessed_bank_train_df.columns[:-1]),
                    preprocessed_bank_train_df['y'], 16, T, 6)


fig_rf1 = plt.figure(1)
fig_rf1.suptitle('Random Forest Error Set Size 2')
plt.plot(rf_2.train_error, 'b', label='Train Error')
plt.plot(rf_2.test_error, 'r', label='Test Error')
plt.legend()
fig_rf1.savefig('Random Forest (Error) Subset 2.png')


fig_rf2 = plt.figure(2)
fig_rf2.suptitle('Random Forest Error Set Size 4')
plt.plot(rf_4.train_error, 'b', label='Train Error')
plt.plot(rf_4.test_error, 'r', label='Test Error')
plt.legend()
fig_rf2.savefig('Random Forest (Error) Subset 4.png')


fig_rf3 = plt.figure(3)
fig_rf3.suptitle('Random Forest Error Set Size 6')
plt.plot(rf_6.train_error, 'b', label='Train Error')
plt.plot(rf_6.test_error, 'r', label='Test Error')
plt.legend()
fig_rf3.savefig('Random Forest (Error) Subset 6.png')

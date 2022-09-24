import pandas as pd

from DecisionTree.preprocessing import preprocessing_bank_data, replace_unknown_data
from ID3_algo import Decision_Tree

# car data training and testing

car_train_data = pd.read_csv('car/train.csv')
car_test_data = pd.read_csv('car/test.csv')

car_test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
car_train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

# create dataframe of car for latex table
df = pd.DataFrame(columns=["Depth", "Entropy_train", "Entropy_test", "Gini_train", "Gini_test", "Major_train", "Major_test"])

# build the decision tree
for depth in range(1, 7):
    for benchmark in ['entropy', 'gini', 'majority']:
        car_decision_tree = Decision_Tree(car_train_data, list(car_train_data.columns[:-1]), car_train_data['label'],
                                         max_depth=depth)

        # get train and  test errors for dataframe
        if benchmark == 'entropy':
            entropy_train = car_decision_tree.training_error('label')
            entropy_test = car_decision_tree.evaluate(car_test_data, 'label')

        elif benchmark == 'gini':
            gini_train = car_decision_tree.training_error('label')
            gini_test = car_decision_tree.evaluate(car_test_data, 'label')

        else:
            major_train = car_decision_tree.training_error('label')
            major_test = car_decision_tree.evaluate(car_test_data, 'label')

        print(f"Car Dataset Errors:")
        print(f"Depth:{depth} Benchmark:{benchmark} =>")
        print(f"Average prediction training error: {car_decision_tree.training_error('label')}")
        print(f"Average prediction testing error: {car_decision_tree.evaluate(car_test_data, 'label')}\n")

    # insert data into the dataframe
    df.loc[len(df)] = [depth, entropy_train, entropy_test, gini_train, gini_test, major_train, major_test]

# create latex table code from dataframe of car
print(df.to_latex())

# bank data training and testing
bank_train_data = pd.read_csv('bank/train.csv')
bank_test_data = pd.read_csv('bank/test.csv')
bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

bank_train_data.columns = bank_column_names
bank_test_data.columns = bank_column_names

# bank data numerical columns
bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
# median of numerical attributes of bank train data
numerical_thresholds = bank_train_data[bank_numerical_columns].median()

#  consider unknown as category
preprocessed_bank_train_df = preprocessing_bank_data(bank_train_data, numerical_thresholds, bank_numerical_columns)
preprocessed_bank_test_df = preprocessing_bank_data(bank_test_data, numerical_thresholds, bank_numerical_columns)

# create dataframe of bank for latex table
df1 = pd.DataFrame(columns=["Depth", "Entropy_train", "Entropy_test", "Gini_train", "Gini_test", "Major_train", "Major_test"])

print("Bank Dataset Evaluation (with unknown considered as value):")
# build the decision tree
for depth in range(1, 17):
    for benchmark in ['entropy', 'gini', 'majority']:
        bank_decision_tree = Decision_Tree(preprocessed_bank_train_df, list(preprocessed_bank_train_df.columns[:-1]),
                                          preprocessed_bank_train_df['y'], max_depth=depth)

        # get train and  test errors for dataframe
        if benchmark == 'entropy':
            entropy_train = bank_decision_tree.training_error('y')
            entropy_test = bank_decision_tree.evaluate(preprocessed_bank_test_df, 'y')

        elif benchmark == 'gini':
            gini_train = bank_decision_tree.training_error('y')
            gini_test = bank_decision_tree.evaluate(preprocessed_bank_test_df, 'y')

        else:
            major_train = bank_decision_tree.training_error('y')
            major_test = bank_decision_tree.evaluate(preprocessed_bank_test_df, 'y')

        print(f"Bank Dataset Errors:")
        print(f"Depth:{depth} Benchmark:{benchmark} =>")
        print(f"Average prediction training error: {bank_decision_tree.training_error('y')}")
        print(f"Average prediction testing error: {bank_decision_tree.evaluate(preprocessed_bank_test_df, 'y')}\n")


    df1.loc[len(df1)] = [depth, entropy_train, entropy_test, gini_train, gini_test, major_train, major_test]

# create latex table code from dataframe of bank
print(df1.to_latex())

# categorical columns with value unknown
categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

# replace unknown by most frequent value
preprocessed_bank_train_df = replace_unknown_data(bank_train_data, categorical_columns_with_unknown_values)
preprocessed_bank_test_df = replace_unknown_data(bank_test_data, categorical_columns_with_unknown_values)


# create dataframe of bank for replaced unknown values for latex table
df2 = pd.DataFrame(columns=["Depth", "Entropy_train", "Entropy_test", "Gini_train", "Gini_test", "Major_train", "Major_test"])

print("Bank Dataset Evaluation (with unknown replaced by most frequent value):")
# build the decision tree
for depth in range(1, 17):
    for benchmark in ['entropy', 'gini', 'majority']:
        bank_decision_tree_for_replaced_unknown_values = Decision_Tree(preprocessed_bank_train_df,
                                                                      list(preprocessed_bank_train_df.columns[:-1]),
                                                                      preprocessed_bank_train_df['y'], max_depth=depth)

        # get train and  test errors for dataframe
        if benchmark == 'entropy':
            entropy_train = bank_decision_tree_for_replaced_unknown_values.training_error('y')
            entropy_test = bank_decision_tree_for_replaced_unknown_values.evaluate(preprocessed_bank_test_df, 'y')

        elif benchmark == 'gini':
            gini_train = bank_decision_tree_for_replaced_unknown_values.training_error('y')
            gini_test = bank_decision_tree_for_replaced_unknown_values.evaluate(preprocessed_bank_test_df, 'y')

        else:
            major_train = bank_decision_tree_for_replaced_unknown_values.training_error('y')
            major_test = bank_decision_tree_for_replaced_unknown_values.evaluate(preprocessed_bank_test_df, 'y')

        print(f"Bank Dataset (replaced unknown value) Errors:")
        print(f"Depth:{depth} Benchmark:{benchmark} =>")
        print(f"Average prediction training error: {bank_decision_tree_for_replaced_unknown_values.training_error('y')}")
        print(f"Average prediction testing error: {bank_decision_tree_for_replaced_unknown_values.evaluate(preprocessed_bank_test_df, 'y')}\n")

    df2.loc[len(df2)] = [depth, entropy_train, entropy_test, gini_train, gini_test, major_train, major_test]

# create latex table code from dataframe of bank for replaced unknown values
print(df2.to_latex())
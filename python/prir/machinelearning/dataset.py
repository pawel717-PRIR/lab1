import pandas as pd


def read_train_data():
    data = pd.read_csv(filepath_or_buffer=train_data_file_path, header=None)
    data.rename(columns={0: 'target'}, inplace=True)
    return data


def read_test_data():
    data = pd.read_csv(filepath_or_buffer=train_data_file_path, header=None)
    data.rename(columns={0: 'target'}, inplace=True)
    return data


def get_target(data):
    data['target'].as_matrix()


def drop_labels(data):
    return data.drop('label', axis=1)


train_data_file_path = "../dataset/mnist_train.csv"
test_data_file_path = "../dataset/mnist_test.csv"









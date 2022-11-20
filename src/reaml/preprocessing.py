import os
import pickle


def setup_train_test_idx(X, last_train_time_step, last_time_step, aggregated_timestamp_column='time_step'):
    split_timesteps = {'train': list(range(last_train_time_step + 1)),
                       'test': list(range(last_train_time_step + 1, last_time_step + 1))}

    train_test_idx = {'train': X[X[aggregated_timestamp_column].isin(split_timesteps['train'])].index,
                      'test': X[X[aggregated_timestamp_column].isin(split_timesteps['test'])].index}

    return train_test_idx


def train_test_split(X, y, train_test_idx):
    X_train_df = X.loc[train_test_idx['train']]
    X_test_df = X.loc[train_test_idx['test']]

    y_train = y.loc[train_test_idx['train']]
    y_test = y.loc[train_test_idx['test']]

    return X_train_df, X_test_df, y_train, y_test


def save_as_pkl(input, PKL_PATH):
    output = open(os.path.join(os.getcwd(), PKL_PATH), 'wb')
    pickle.dump(input, output)
    output.close()

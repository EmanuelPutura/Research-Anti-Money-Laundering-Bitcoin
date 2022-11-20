import os
import pandas as pd

from reaml.preprocessing import setup_train_test_idx, train_test_split


ROOT_DIR = os.getcwd()


def insertAdditionalFeatures1(df_classes, df_edges, df_features):
    dict_classes = dict(zip(df_classes.txId, df_classes['class']))
    dict_edges = dict(zip(df_edges.txId1, df_edges.txId2))

    illicit_neighbours = {}

    for source_node, dest_node in dict_edges.items():
      if dict_classes[source_node] == '1':
        if dest_node not in illicit_neighbours:
          illicit_neighbours[dest_node] = 0
        
        if source_node not in illicit_neighbours:
          illicit_neighbours[source_node] = 0

        illicit_neighbours[dest_node] += 1
        illicit_neighbours[source_node] += 1

    df_illicit_neighbours = pd.DataFrame(illicit_neighbours.items(), columns=['Node', 'Illicit'])
    df_features = pd.merge(df_features, df_illicit_neighbours[['Node', 'Illicit']], left_on=0, right_on='Node', how='left')
    df_features['Illicit'] = df_features['Illicit'].fillna(0)

    return df_features.drop('Node', axis=1)


def insertAdditionalFeatures2(df_classes, df_edges, df_features):
    dict_classes = dict(zip(df_classes.txId, df_classes['class']))
    dict_edges = dict(zip(df_edges.txId1, df_edges.txId2))

    dest_illicit_neighbours = {}
    source_illicit_neighbours = {}

    for source_node, dest_node in dict_edges.items():
      if dict_classes[source_node] == '1':
        if dest_node not in dest_illicit_neighbours:
          dest_illicit_neighbours[dest_node] = 0
        
        if source_node not in source_illicit_neighbours:
          source_illicit_neighbours[source_node] = 0

        dest_illicit_neighbours[dest_node] += 1
        source_illicit_neighbours[source_node] += 1

    df_dest_illicit_neighbours = pd.DataFrame(dest_illicit_neighbours.items(), columns=['Node', 'Dest Illicit'])
    df_features = pd.merge(df_features, df_dest_illicit_neighbours[['Node', 'Dest Illicit']], left_on=0, right_on='Node', how='left')
    df_features['Dest Illicit'] = df_features['Dest Illicit'].fillna(0)
    df_features = df_features.drop('Node', axis=1)

    df_source_illicit_neighbours = pd.DataFrame(source_illicit_neighbours.items(), columns=['Node', 'Source Illicit'])
    df_features = pd.merge(df_features, df_source_illicit_neighbours[['Node', 'Source Illicit']], left_on=0, right_on='Node', how='left')
    df_features['Source Illicit'] = df_features['Source Illicit'].fillna(0)
    df_features = df_features.drop('Node', axis=1)

    return df_features


def import_elliptic_data_from_csvs():
    df_classes = pd.read_csv(os.path.join(ROOT_DIR, 'reaml/data/elliptic/elliptic_txs_classes.csv'))
    df_edges = pd.read_csv(os.path.join(ROOT_DIR, 'reaml/data/elliptic/elliptic_txs_edgelist.csv'))
  
    df_features = pd.read_csv(os.path.join(ROOT_DIR, 'reaml/data/elliptic/elliptic_txs_features.csv'), header=None)
    df_features = insertAdditionalFeatures2(df_classes, df_edges, df_features)

    return df_classes, df_edges, df_features


def calc_occurences_per_timestep():
    X, y = load_elliptic_data()
    X['class'] = y
    occ = X.groupby(['time_step', 'class']).size().to_frame(name='occurences').reset_index()
    return occ


def rename_classes(df_classes):
    df_classes.replace({'class': {'1': 1, '2': 0, 'unknown': 2}}, inplace=True)
    return df_classes


def rename_features(df_features):
    # TODO: modify values
    df_features.columns = ['id', 'time_step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                          range(74)]
    return df_features


def import_and_clean_elliptic_data():
    df_classes, df_edges, df_features = import_elliptic_data_from_csvs()
    df_classes = rename_classes(df_classes)
    df_features = rename_features(df_features)
    return df_classes, df_edges, df_features


def combine_dataframes(df_classes, df_features, only_labeled=True):
    df_combined = pd.merge(df_features, df_classes, left_on='id', right_on='txId', how='left')
    if only_labeled:
        df_combined = df_combined[df_combined['class'] != 2].reset_index(drop=True)
    df_combined.drop(columns=['txId'], inplace=True)
    return df_combined


def import_elliptic_edgelist():
    df_classes, df_edges, df_features = import_and_clean_elliptic_data()
    df_edgelist = df_edges.merge(df_features[['id', 'time_step']], left_on='txId1', right_on='id')
    return df_edgelist


def load_elliptic_data(only_labeled=True, drop_node_id=True):
    df_classes, df_edges, df_features = import_elliptic_data_from_csvs()
    df_features = rename_features(df_features)
    df_classes = rename_classes(df_classes)
    df_combined = combine_dataframes(df_classes, df_features, only_labeled)

    if drop_node_id:
        X = df_combined.drop(columns=['id', 'class'])
    else:
        X = df_combined.drop(columns='class')

    y = df_combined['class']

    return X, y


def run_elliptic_preprocessing_pipeline(last_train_time_step, last_time_step, only_labeled=True,
                                        drop_node_id=True):
    X, y = load_elliptic_data(only_labeled, drop_node_id)
    train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)

    return X_train_df, X_test_df, y_train, y_test

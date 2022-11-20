import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def calculate_model_score(y_true, y_pred, metric):
    metric_dict = {'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, pos_label=1),
                   'f1_micro': f1_score(y_true, y_pred, average='micro'),
                   'f1_macro': f1_score(y_true, y_pred, average='macro'),
                   'precision': precision_score(y_true, y_pred), 'recall': recall_score(y_true, y_pred),
                   'roc_auc': roc_auc_score(y_true, y_pred)}
    model_score = metric_dict[metric]
    return model_score


def calc_average_score(y_test, y_preds, scoring='f1'):
    all_model_scores = []
    for y_pred in y_preds:
        model_score = calculate_model_score(y_test.astype('int'), y_pred, scoring)
        all_model_scores.append(model_score)

    avg = np.mean(all_model_scores)

    return avg


def calc_average_score_and_std_per_timestep(X_test_df, y_test, y_preds, aggregated_timestamp_column='time_step', scoring='f1'):
    last_train_time_step = min(X_test_df['time_step']) - 1
    last_time_step = max(X_test_df['time_step'])
    all_model_scores = []
    for y_pred in y_preds:
        model_scores = []
        for time_step in range(last_train_time_step + 1, last_time_step + 1):
            time_step_idx = np.flatnonzero(X_test_df[aggregated_timestamp_column] == time_step)
            y_true_ts = y_test.iloc[time_step_idx]
            y_pred_ts = [y_pred[i] for i in time_step_idx]
            model_scores.append(calculate_model_score(y_true_ts.astype('int'), y_pred_ts, scoring))
        all_model_scores.append(model_scores)

    avg_f1 = np.array([np.mean([f1_scores[i] for f1_scores in all_model_scores]) for i in range(15)])
    std = np.array([np.std([f1_scores[i] for f1_scores in all_model_scores]) for i in range(15)])

    return avg_f1, std

import os
import csv

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from experiments.general_functions.elliptic_data_preprocessing import run_elliptic_preprocessing_pipeline
from reaml.models import supervised_model_cv_fit_predict
from reaml.model_performance import calc_average_score_and_std_per_timestep, calc_average_score
from reaml.preprocessing import save_as_pkl


ROOT_DIR = os.getcwd()


def evaluate_supervised_model():
    # Import Elliptic data set and set variables
    last_time_step = 49
    last_train_time_step = 34
    only_labeled = True

    X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(
        last_train_time_step=last_train_time_step,
        last_time_step=last_time_step,
        only_labeled=only_labeled)

    y_preds_xgb = supervised_model_cv_fit_predict(X_train_df, y_train, X_test_df, XGBClassifier())
    avg_f1_xgb_ts, std_xgb_ts = calc_average_score_and_std_per_timestep(X_test_df, y_test, y_preds_xgb)
    avg_f1_xgb = calc_average_score(y_test, y_preds_xgb)

    y_preds_rf = supervised_model_cv_fit_predict(X_train_df, y_train, X_test_df, RandomForestClassifier())
    avg_f1_rf_ts, std_rf_ts = calc_average_score_and_std_per_timestep(X_test_df, y_test, y_preds_rf)
    avg_f1_rf = calc_average_score(y_test, y_preds_rf)

    y_preds_lr = supervised_model_cv_fit_predict(X_train_df, y_train, X_test_df, LogisticRegression(max_iter=10000))
    avg_f1_lr_ts, std_lr_ts = calc_average_score_and_std_per_timestep(X_test_df, y_test, y_preds_lr)
    avg_f1_lr = calc_average_score(y_test, y_preds_lr)

    model_f1_ts_dict = {'XGBoost': avg_f1_xgb_ts, 'Logistic Regression': avg_f1_lr_ts, 'Random Forest': avg_f1_rf_ts}
    model_std_ts_dict = {'XGBoost': std_xgb_ts, 'Logistic Regression': std_lr_ts, 'Random Forest': std_rf_ts}

    # Save average f1 results of all classifiers over 5 runs
    model_avg_f1_dict = {'XGBoost': avg_f1_xgb, 'Random Forest': avg_f1_rf,
                         'Logistic Regression': avg_f1_lr}

    with open(os.path.join(ROOT_DIR, 'reaml/output/supervised_avg_f1_per_model_over_5_runs.csv'), 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in model_avg_f1_dict.items():
            writer.writerow([key, value])

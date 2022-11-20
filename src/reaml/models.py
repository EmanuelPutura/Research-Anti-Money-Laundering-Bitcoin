import numpy as np
import random


def supervised_model_cv_fit_predict(X_train_df, y_train, X_test_df, model, runs=5):
    y_preds = []

    for i in range(runs):
        random.seed(i)
        model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)
        y_preds.append(y_pred)

    return y_preds

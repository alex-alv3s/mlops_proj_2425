import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os
import pickle
from sklearn.linear_model import LassoCV

def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> list:
    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")

    columns_to_exclude = ['id']
    X_train = X_train.drop(columns=columns_to_exclude, errors='ignore')

    valid_methods = {"rfe", "lasso", "rfe+lasso"}
    method = parameters.get("feature_selection", "").lower()
    if method not in valid_methods:
        raise ValueError(f"Invalid feature_selection method: '{method}'. Must be one of {valid_methods}.")

    y_train = np.ravel(y_train)
    X_cols_rfe = X_train.columns.tolist()
    X_cols_lasso = X_train.columns.tolist()

    if method == "rfe":
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
        except:
            classifier = RandomForestClassifier(**parameters['baseline_model_params_random_forest'])
        rfe = RFE(classifier)
        rfe.fit(X_train, y_train)
        X_cols_rfe = X_train.columns[rfe.get_support()].tolist()

    if method == "lasso":
        lasso = LassoCV(cv=5, random_state=parameters.get("random_state", 42))
        lasso.fit(X_train, y_train)
        coef_mask = lasso.coef_ != 0
        X_cols_lasso = X_train.columns[coef_mask].tolist()

    if method == "rfe+lasso":
        classifier = RandomForestClassifier(**parameters['baseline_model_params_random_forest'])
        rfe = RFE(classifier)
        rfe.fit(X_train, y_train)
        X_cols_rfe = X_train.columns[rfe.get_support()].tolist()
        lasso = LassoCV(cv=5, random_state=parameters.get("random_state", 42))
        lasso.fit(X_train, y_train)
        coef_mask = lasso.coef_ != 0
        X_cols_lasso = X_train.columns[coef_mask].tolist()
        X_cols = list(set(X_cols_rfe) & set(X_cols_lasso))
    else:
        X_cols = X_cols_rfe if parameters["feature_selection"] == "rfe" else X_cols_lasso

    log.info(f"Number of selected columns: {len(X_cols)}")
    return X_cols
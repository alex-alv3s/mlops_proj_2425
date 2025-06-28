import pandas as pd
import logging
from typing import Dict, Tuple, Any
import pickle
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import mlflow
import numpy as np



logger = logging.getLogger(__name__)

def model_predict(
    parameters: Dict[str, Any],
    X: pd.DataFrame,
    model: pickle.Pickler,
    columns: list
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Use the trained model to predict on new data.

    Args:
        X (pd.DataFrame): Input features for prediction.
        model (pickle): Trained model object.
        columns (list): List of feature columns expected by the model.

    Returns:
        Tuple containing:
          - DataFrame with predictions added as 'y_pred' column.
          - Dictionary with descriptive stats of the predictions dataframe.
    """

    target_column = parameters['target_column']
    X = X.drop(columns=['id'], errors='ignore')
    
    if target_column:
        if target_column not in X.columns:
            logger.warning(f"Target column '{target_column}' not found in input data. Predictions will not include this column.")
        else:
            logger.warning(f"Target column '{target_column}' found in input data. Predictions will include this column.")
    else:
        logger.warning("No target column specified. Predictions will not include a target column.")

    for col in columns:
        if col not in X.columns:
            X[col] = 0 

    X_selected = X[columns]

    y_pred = model.predict(X_selected)

    X = X.copy()
    X['y_pred'] = y_pred

    describe_servings = X.describe().to_dict()

    #note:
    #since this is an academic project the target column resides in the batch data, therefore we can evaluate the production model
    #under normal circumstances this wouln't be possible as the target column would not be available in production data
    if target_column and target_column in X.columns:
        logger.info(f"Evaluating model performance using target column '{target_column}'.")
        y_true = X[target_column]
        metrics = {
            "f1_score": f1_score(y_true, y_pred, average="binary"),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        preds_output_path = "predictions.json"
        X[['y_pred']].to_json(preds_output_path, orient="records", lines=True)
        mlflow.log_artifact(preds_output_path)

        describe_servings.update(metrics)
    else:
        logger.warning("Target column not found in predictions, skipping evaluation metrics.")

    logger.info(f"Predicted {len(y_pred)} records.")

    return X, describe_servings

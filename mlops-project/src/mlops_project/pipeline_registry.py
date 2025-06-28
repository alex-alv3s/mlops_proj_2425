"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from mlops_project.pipelines import (
    data_unit_tests as data_tests,
    preprocess_train as preprocess_train,
    preprocess_batch as preprocessing_batch,
    feature_selection as feature_selection_pipeline,
    split_train as split_train,
    split_data as split_data,
    model_selection as model_selection_pipeline,
    model_predict as model_predict,
    model_train as model_train_pipeline,
    feature_store as feature_store_pipeline,
    data_drift as data_drift_pipeline,
    drift_report as drift_report_pipeline,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_unit_tests_pipeline = data_tests.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    preprocess_batch_pipeline = preprocessing_batch.create_pipeline()
    feature_selection = feature_selection_pipeline.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    model_selection = model_selection_pipeline.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()
    model_train = model_train_pipeline.create_pipeline()
    feature_store = feature_store_pipeline.create_pipeline()
    data_drift = data_drift_pipeline.create_pipeline()
    drift_report = drift_report_pipeline.create_pipeline()

    return {
        "data_unit_tests": data_unit_tests_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "preprocess_batch": preprocess_batch_pipeline,
        "feature_selection": feature_selection,
        "split_train": split_train_pipeline,
        "split_data": split_data_pipeline,
        "model_selection": model_selection,
        "model_predict": model_predict_pipeline,
        "model_train": model_train,
        "feature_store": feature_store,
        "data_drift": data_drift,
        "drift_report": drift_report
    }
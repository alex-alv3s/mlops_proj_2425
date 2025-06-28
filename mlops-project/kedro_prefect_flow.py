from prefect import flow, task, get_run_logger
from src.mlops_project.run_kedro_pipeline import run_pipeline
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

@task
def run_kedro_task(pipeline_name: str):
    logger = get_run_logger()
    logger.info(f"Running Kedro pipeline: `{pipeline_name}`")

    try:
        run_pipeline(pipeline_name)
        logger.info(f"Kedro pipeline `{pipeline_name}` finished successfully.")
    except Exception as e:
        logger.error(f"Pipeline `{pipeline_name}` failed: {str(e)}")
        raise

@flow(name="Data Unit Test Flow", description="Runs data unit tests")
def flow_data_unit_tests():
    run_kedro_task("split_data")
    run_kedro_task("data_unit_tests")

@flow(name="Full Training Processing Flow", description="Runs full preprocessing pipeline")
def flow_full_processing():
    run_kedro_task("preprocess_train")
    run_kedro_task("preprocess_batch")
    run_kedro_task("feature_store")
    run_kedro_task("split_train")

@flow(name="Model Training Flow", description="Runs model training and selection as well as data drift detection")
def flow_train():
    run_kedro_task("feature_selection")
    run_kedro_task("model_train")
    run_kedro_task("model_selection")
    run_kedro_task("model_train")
    run_kedro_task("data_drift")

@flow(name="Model Predicting Flow", description="Runs model prediction on batch data")
def flow_deploy():
    run_kedro_task("drift_report")
    run_kedro_task("model_predict")

@flow(name="Orchestration Flow", description="Runs the entire Kedro pipeline from data unit tests to deployment")
def full_pipeline():
    flow_data_unit_tests()
    flow_full_processing() 
    flow_train()
    flow_deploy()

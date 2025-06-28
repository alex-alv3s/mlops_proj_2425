from prefect.client.schemas.schedules import CronSchedule
from kedro_prefect_flow import flow_data_unit_tests, full_pipeline, flow_full_processing, flow_train, flow_deploy
from prefect import serve

def serve_all_flows():
    """Serve all flows - they'll be available in the UI"""
    
    print("Starting Prefect deployment server...")
    
    serve(
        flow_data_unit_tests.to_deployment(
            name="data-unit-tests",
            tags=["unit-tests", "on-demand"],
            description="Run data unit tests to validate data quality"
        ),
        flow_full_processing.to_deployment(
            name="full-processing", 
            tags=["training-preprocess", "on-demand"],
            description="Complete data processing pipeline"
        ),
        flow_train.to_deployment(
            name="model-training",
            tags=["training", "on-demand"],
            description="Train machine learning models"
        ),
        flow_deploy.to_deployment(
            name="model-deployment",
            tags=["deployment", "on-demand"],
            description="Deploy trained models to production"
        ),
        full_pipeline.to_deployment(
            name="full-pipeline",
            tags=["full-pipeline", "on-demand"],
            description="Complete end-to-end ML pipeline"
        )
    )

if __name__ == "__main__":
    serve_all_flows()
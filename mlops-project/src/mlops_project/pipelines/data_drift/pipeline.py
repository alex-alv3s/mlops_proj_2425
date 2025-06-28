
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import drift_detection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= drift_detection,
                inputs=["preprocessed_batch_data", "preprocessed_train_data"],
                outputs= ["drift_metrics", "drift_flags", "drift_report"],
                name = "data_drift",
            ),

        ]
    )

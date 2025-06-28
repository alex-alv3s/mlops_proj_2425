"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=["parameters", "preprocessed_batch_data", "production_model", "production_columns"],
                outputs=["predicted_data", "predict_describe"],
                name="predict_node",
            ),
        ]
    )


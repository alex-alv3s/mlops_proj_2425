
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_store

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= feature_store,
                inputs=["preprocessed_training_data","parameters"],
                outputs= "preprocessed_train_data",
                name="feature_store",
            ),

        ]
    )

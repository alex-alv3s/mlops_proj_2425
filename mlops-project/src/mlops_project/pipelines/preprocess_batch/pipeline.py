
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            node(
                func= clean_data,
                inputs="ana_data",
                outputs= "ana_data_cleaned",
                name="clean_data",
            ),
            node(
                func= feature_engineer,
                inputs=["ana_data_cleaned","encoder_transform"],
                outputs= "preprocessed_batch_data",
                name="preprocessed_batch",
            )
        ]
    )
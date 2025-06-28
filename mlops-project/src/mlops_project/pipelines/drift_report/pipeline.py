from kedro.pipeline import Pipeline, node, pipeline
from .nodes import drift_report_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=drift_report_node,
                inputs=["drift_flags", "drift_report"],
                outputs="drift_report_summary",
                name="drift_report_node",
            ),
        ]
    )

"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration


from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings


import hopsworks

import great_expectations as ge


logger = logging.getLogger(__name__)

from great_expectations.core import ExpectationSuite, ExpectationConfiguration

def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Creates Great Expectations suites for stroke dataset features.

    Args:
        expectation_suite_name (str): Name of the expectation suite.
        feature_group (str): Group to define expectations for. Can be 'numerical', 'one_hot', or 'target'.

    Returns:
        ExpectationSuite: Configured expectation suite for the group.
    """
    suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)

    if feature_group == "numerical_features":
        for col in ["age", "avg_glucose_level", "bmi"]:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "float64"}
                )
            )
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col}
                )
            )

        for col in ["hypertension", "heart_disease", "is_urban"]:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={"column": col, "value_set": [0, 1]}
                )
            )

    elif feature_group == "categorical_features":
        one_hot_columns = [
            "oh_gender_female", "oh_gender_male", "oh_gender_other",
            "oh_work_type_govt_job", "oh_work_type_never_worked", "oh_work_type_private",
            "oh_work_type_self_employed", "oh_work_type_children",
            "oh_smoking_status_unknown", "oh_smoking_status_formerly_smoked",
            "oh_smoking_status_never_smoked", "oh_smoking_status_smokes",
            "oh_bmi_cat_ideal", "oh_bmi_cat_obesity", "oh_bmi_cat_overweight", "oh_bmi_cat_underweight",
            "oh_age_cat_adults", "oh_age_cat_children", "oh_age_cat_elderly",
            "oh_age_cat_mid_adults", "oh_age_cat_teens",
            "oh_glucose_cat_high", "oh_glucose_cat_low", "oh_glucose_cat_normal", "oh_glucose_cat_very_high"
        ]
        for col in one_hot_columns:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={"column": col, "value_set": [0, 1]}
                )
            )

    elif feature_group == "target":
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "stroke", "value_set": [0, 1]}
            )
        )

    return suite

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes a DataFrame and a Great Expectations ExpectationSuite,
    validates the data against the suite, then saves the data to the feature store.

    Args:
        data (pd.DataFrame): Dataframe with data to store.
        group_name (str): Feature group name.
        feature_group_version (int): Feature group version.
        description (str): Description of the feature group.
        group_description (dict): Descriptions of each feature.
        validation_expectation_suite (ExpectationSuite): Great Expectations suite.
        credentials_input (dict): Credentials for hopsworks feature store connection.

    Returns:
        object_feature_group: The feature group object in the feature store.
    """

    ge_df = ge.from_pandas(data)

    validation_result = ge_df.validate(expectation_suite=validation_expectation_suite, result_format="SUMMARY")

    if not validation_result.success:
        raise ValueError(f"Data validation failed for feature group '{group_name}'. Errors: {validation_result}")

    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], 
        project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["id"],
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    print(f"Data columns for feature group '{group_name}': {data.columns}")
    print(f"Are columns unique? {data.columns.is_unique}")
    print("Data column types:")
    print(data.dtypes)
    for col in data.columns:
        print(col, type(data[col]), getattr(data[col], 'shape', None))

    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    for desc in group_description:
        object_feature_group.update_feature_description(
            desc["name"], desc["description"]
        )

    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def feature_store(
    df1: pd.DataFrame,
    parameters: Dict[str, Any]
):
    """
    Prepare dataset, validate with Great Expectations, and optionally store features in Hopsworks Feature Store.

    Args:
        df1 (pd.DataFrame): Preprocessed DataFrame including all features and target.
        parameters (dict): Includes 'target_column' and 'to_feature_store'.

    Returns:
        pd.DataFrame: Original input dataframe.
    """

    conf_path = str(Path('') / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    credentials = conf_loader["credentials"]

    logger.info(f"The dataset contains {len(df1.columns)} columns.")

    numerical_features = [col for col in df1.columns if not col.startswith("oh_") and col not in ["id", parameters["target_column"]]]
    categorical_features = [col for col in df1.columns if col.startswith("oh_")]

    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations", "numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations", "categorical_features")
    validation_expectation_suite_target = build_expectation_suite("target_expectations", "target")

    numerical_feature_descriptions = [
        {"name": "age", "description": "Age of the patient in years"},
        {"name": "avg_glucose_level", "description": "Average blood glucose level"},
        {"name": "bmi", "description": "Body Mass Index"},
        {"name": "hypertension", "description": "1 if the patient has hypertension, else 0"},
        {"name": "heart_disease", "description": "1 if the patient has heart disease, else 0"},
        {"name": "is_urban", "description": "1 if residence is urban, 0 if rural"}
    ]

    categorical_feature_descriptions = [
        {"name": "oh_gender_female", "description": "1 if gender is female, else 0"},
        {"name": "oh_gender_male", "description": "1 if gender is male, else 0"},
        {"name": "oh_gender_other", "description": "1 if gender is other, else 0"},
        {"name": "oh_work_type_govt_job", "description": "1 if work type is government job, else 0"},
        {"name": "oh_work_type_never_worked", "description": "1 if never worked, else 0"},
        {"name": "oh_work_type_private", "description": "1 if work type is private, else 0"},
        {"name": "oh_work_type_self_employed", "description": "1 if self-employed, else 0"},
        {"name": "oh_work_type_children", "description": "1 if child (never worked), else 0"},
        {"name": "oh_smoking_status_unknown", "description": "1 if smoking status is unknown, else 0"},
        {"name": "oh_smoking_status_formerly_smoked", "description": "1 if formerly smoked, else 0"},
        {"name": "oh_smoking_status_never_smoked", "description": "1 if never smoked, else 0"},
        {"name": "oh_smoking_status_smokes", "description": "1 if currently smokes, else 0"},
        {"name": "oh_bmi_cat_ideal", "description": "1 if BMI is ideal, else 0"},
        {"name": "oh_bmi_cat_obesity", "description": "1 if BMI indicates obesity, else 0"},
        {"name": "oh_bmi_cat_overweight", "description": "1 if BMI indicates overweight, else 0"},
        {"name": "oh_bmi_cat_underweight", "description": "1 if BMI indicates underweight, else 0"},
        {"name": "oh_age_cat_adults", "description": "1 if patient is an adult, else 0"},
        {"name": "oh_age_cat_children", "description": "1 if patient is a child, else 0"},
        {"name": "oh_age_cat_elderly", "description": "1 if patient is elderly, else 0"},
        {"name": "oh_age_cat_mid_adults", "description": "1 if patient is a middle-aged adult, else 0"},
        {"name": "oh_age_cat_teens", "description": "1 if patient is a teenager, else 0"},
        {"name": "oh_glucose_cat_high", "description": "1 if glucose level is high, else 0"},
        {"name": "oh_glucose_cat_low", "description": "1 if glucose level is low, else 0"},
        {"name": "oh_glucose_cat_normal", "description": "1 if glucose level is normal, else 0"},
        {"name": "oh_glucose_cat_very_high", "description": "1 if glucose level is very high, else 0"}
    ]

    target_feature_descriptions = [
        {"name": "stroke", "description": "1 if patient had a stroke, else 0"}
    ]

    df1_numeric = df1[["id"] + numerical_features]
    df1_categorical = df1[["id"] + categorical_features]
    df1_target = df1[["id", parameters["target_column"]]]

    if parameters.get("to_feature_store", False):
        try:
            to_feature_store(
                df1_numeric,
                "numerical_features",
                1,
                "Numerical Features",
                numerical_feature_descriptions,
                validation_expectation_suite_numerical,
                credentials["feature_store"]
            )
        except ValueError as e:
            logger.error(f"Failed to store numerical features: {e}")
        try:
            to_feature_store(
                df1_categorical,
                "categorical_features",
                1,
                "Categorical Features",
                categorical_feature_descriptions,
                validation_expectation_suite_categorical,
                credentials["feature_store"]
            )
        except ValueError as e:
            logger.error(f"Failed to store categorical features: {e}")
        try:
            to_feature_store(
                df1_target,
                "target_feature",
                1,
                "Target Feature",
                target_feature_descriptions,
                validation_expectation_suite_target,
                credentials["feature_store"]
            )
        except ValueError as e:
            logger.error(f"Failed to store target feature: {e}")
    return df1


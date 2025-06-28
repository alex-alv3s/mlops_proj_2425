"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

from pathlib import Path

# from kedro.config import OmegaConfigLoader
# from kedro.framework.project import settings

import os


logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))
    validation_result_ = validation_result_data.get('validation_result', {})
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if type(observed_value) is list:
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation


def test_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    base_dir = os.getcwd()
    target_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'gx'))
    context = gx.get_context(context_root_dir = target_path)
    datasource_name = "stroke_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_stroke = context.add_or_update_expectation_suite(expectation_suite_name="expectation_suite")
    
    hypertension_values = sorted(train_df['hypertension'].dropna().unique().tolist())
    heart_disease_values = sorted(train_df['heart_disease'].dropna().unique().tolist())
    ever_married_values = sorted(train_df['ever_married'].dropna().unique().tolist())
    work_type_values = sorted(train_df['work_type'].dropna().unique().tolist())
    residence_type_values = sorted(train_df['Residence_type'].dropna().unique().tolist())
    smoking_status_values = sorted(train_df['smoking_status'].dropna().unique().tolist())
    gender_values = sorted(train_df['gender'].dropna().unique().tolist())
    
    age_min = train_df['age'].min()
    age_max = train_df['age'].max()
    
    bmi_min = max(train_df['bmi'].min() * 0.9, 10)
    bmi_max = min(train_df['bmi'].max() * 1.1, 60)
    
    glucose_min = max(train_df['avg_glucose_level'].min() * 0.9, 40)
    glucose_max = min(train_df['avg_glucose_level'].max() * 1.1, 400)
    
    expectation_hypertension = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "hypertension",
            "value_set": hypertension_values
        },
    )
    
    expectation_heart_disease = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "heart_disease",
            "value_set": heart_disease_values
        },
    )

    expectation_ever_married = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "ever_married",
            "value_set": ever_married_values
        },
    )

    expectation_work_type = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "work_type",
            "value_set": work_type_values
        },
    )

    expectation_Residence_Type = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": 'Residence_type',
            "value_set": residence_type_values
        },
    )

    expectation_smoking_status = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "smoking_status",
            "value_set": smoking_status_values
        },
    )

    expectation_gender = ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "gender",
            "value_set": gender_values
        },
    )

    expectation_age_range = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "age",
            "min_value": age_min,
            "max_value": age_max
        },
    )

    expectation_bmi_range = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "bmi",
            "min_value": bmi_min,
            "max_value": bmi_max,
            "strict_min": False,
            "strict_max": False
        },
    )
    
    expectation_bmi_not_null = ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={
            "column": "bmi"
        },
    )
    
    expectation_bmi_type = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={
            "column": "bmi",
            "type_": "float"
        },
    )

    expectation_glucose_range = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "avg_glucose_level",
            "min_value": glucose_min,
            "max_value": glucose_max,
            "strict_min": False,
            "strict_max": False,
        },
    )

    expectation_glucose_not_null = ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={
            "column": "avg_glucose_level"
        },
    )

    expectation_glucose_type = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={
            "column": "avg_glucose_level",
            "type_": "float"
        },
    )

    suite_stroke.add_expectation(expectation_configuration=expectation_hypertension)
    suite_stroke.add_expectation(expectation_configuration=expectation_heart_disease)
    suite_stroke.add_expectation(expectation_configuration=expectation_ever_married)
    suite_stroke.add_expectation(expectation_configuration=expectation_work_type)
    suite_stroke.add_expectation(expectation_configuration=expectation_Residence_Type)
    suite_stroke.add_expectation(expectation_configuration=expectation_smoking_status)
    suite_stroke.add_expectation(expectation_configuration=expectation_gender)
    suite_stroke.add_expectation(expectation_configuration=expectation_age_range)
    suite_stroke.add_expectation(expectation_configuration=expectation_bmi_range)
    suite_stroke.add_expectation(expectation_configuration=expectation_bmi_not_null)
    suite_stroke.add_expectation(expectation_configuration=expectation_bmi_type)
    suite_stroke.add_expectation(expectation_configuration=expectation_glucose_range)
    suite_stroke.add_expectation(expectation_configuration=expectation_glucose_not_null)
    suite_stroke.add_expectation(expectation_configuration=expectation_glucose_type)

    context.add_or_update_expectation_suite(expectation_suite=suite_stroke)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=test_df)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=test_df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "expectation_suite",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)

    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")

    return df_validation
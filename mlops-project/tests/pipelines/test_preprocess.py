import pandas as pd
import numpy as np
import pytest
from src.mlops_project.pipelines.preprocess_train.nodes import clean_data, feature_engineer

@pytest.fixture
def raw_sample():
    data = {
        "id": list(range(15)),
        "gender": ["Male", "Female", "Other"] * 5,
        "age": [25, 54, 12, 30, 45, 23, 60, 33, 27, 40, 55, 20, 17, 22, 50],
        "hypertension": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "heart_disease": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        "ever_married": ["Yes", "No", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "work_type": ["Private", "Govt_job", "children", "Private", "Govt_job", "Private", "children", "Govt_job", "Private", "Private", "Govt_job", "children", "Private", "Govt_job", "Private"],
        "Residence_type": ["Urban", "Rural", "Rural", "Urban", "Rural", "Urban", "Rural", "Urban", "Urban", "Rural", "Urban", "Rural", "Urban", "Rural", "Urban"],
        "avg_glucose_level": [85, 160, 50, 120, 140, 100, 110, 130, 150, 135, 90, 115, 125, 105, 95],
        "bmi": [22.5, 24.0, 18.3, 27.0, 30.0, 23.0, 26.0, 25.0, 28.0, 22.0, 24.5, 19.0, 21.0, 23.5, 26.5],
        "smoking_status": ["smokes", "never smoked", "Unknown", "smokes", "never smoked", "Unknown", "smokes", "never smoked", "Unknown", "smokes", "never smoked", "Unknown", "smokes", "never smoked", "Unknown"],
        "stroke": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_clean_data_fills_bmi_and_lowercases(raw_sample):
    cleaned_df, desc = clean_data(raw_sample)
    assert cleaned_df['bmi'].isnull().sum() == 0
    assert all(col == col.lower() for col in cleaned_df.columns)
    assert isinstance(desc, dict)
    assert 'bmi' in desc
    assert 'age' in desc

def test_feature_engineer_output(raw_sample):
    cleaned_df, _ = clean_data(raw_sample)
    processed_df, encoder = feature_engineer(cleaned_df, target_column='stroke')
    
    oh_cols = processed_df.columns
    assert any(col.startswith("oh_bmi_cat_") for col in oh_cols), "Missing one-hot encoded bmi_cat columns"
    assert any(col.startswith("oh_age_cat_") for col in oh_cols), "Missing one-hot encoded age_cat columns"
    assert any(col.startswith("oh_glucose_cat_") for col in oh_cols), "Missing one-hot encoded glucose_cat columns"
    assert any(col.startswith("oh_work_type_") for col in oh_cols), "Missing one-hot encoded work_type columns"
    assert any(col.startswith("oh_smoking_status_") for col in oh_cols), "Missing one-hot encoded smoking_status columns"
    
    assert set(processed_df['ever_married'].unique()).issubset({0,1})
    assert 'is_urban' in processed_df.columns
    assert set(processed_df['is_urban'].unique()).issubset({0,1})

    cat_cols = [col for col in processed_df.columns if col.startswith("oh_")]
    assert len(cat_cols) > 0
    assert 'residence_type' not in processed_df.columns
    
    assert 'stroke' in processed_df.columns
    
    counts = processed_df['stroke'].value_counts()
    assert counts.min() > 1
    
    feat_names = encoder.get_feature_names_out()
    assert any("bmi_cat" in f for f in feat_names)
    assert any("age_cat" in f for f in feat_names)
    assert any("glucose_cat" in f for f in feat_names)
    assert any("work_type" in f for f in feat_names)
    assert any("smoking_status" in f for f in feat_names)


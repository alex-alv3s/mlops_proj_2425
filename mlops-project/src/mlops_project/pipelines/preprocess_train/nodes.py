"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from imblearn.over_sampling import SMOTENC


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder

def bmi_(data):
    data['bmi_cat'] = pd.cut(data['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
    return data

def age_(data):
    data['age_cat'] = pd.cut(data['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
    return data

def glucose_(data):
    data['glucose_cat'] = pd.cut(data['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])
    return data

def clean_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Data cleaning function.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    df = data.copy()
    df['bmi'] = df['bmi'].fillna(round (df['bmi'].median(), 2))
    df.isnull().sum()

    df.columns = [col.lower().replace("-", "_") for col in df.columns]
    describe_to_dict_verified = df.describe().to_dict()
    
    return df, describe_to_dict_verified


def feature_engineer(data: pd.DataFrame, target_column: str = "stroke"):
    df = bmi_(data)
    df = age_(df)
    df = glucose_(df)

    df["ever_married"] = df["ever_married"].map({'Yes': 1, 'No': 0})
    df["is_urban"] = df["residence_type"].map({'Urban': 1, 'Rural': 0})
    df = df.drop(columns=['residence_type'])

    categorical_features = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[categorical_features]))

    OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)
    OH_cols.index = df.index
    OH_cols.columns = ["oh_" + col.lower().replace("-", "_").replace(" ", "_") for col in OH_cols.columns]

    num_df = df.drop(categorical_features, axis=1)
    num_df.columns = [col.lower().replace("-", "_").replace(" ", "_") for col in num_df.columns]

    df_final = pd.concat([num_df, OH_cols], axis=1)

    log = logging.getLogger(__name__)
    log.info(f"The final dataframe has {len(df_final.columns)} columns.")

    if target_column in df_final.columns:
        X = df_final.drop(columns=[target_column])
        y = df_final[target_column]
        categorical_indices = [i for i, col in enumerate(X.columns) if col.startswith("oh_")]
        smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        df_final_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_column)], axis=1)
        log.info(f"After SMOTE, class distribution: {y_res.value_counts().to_dict()}")
        return df_final_resampled, OH_encoder
    else:
        log.warning(f"Target column '{target_column}' not found for SMOTE. Returning original dataframe.")
        return df_final, OH_encoder
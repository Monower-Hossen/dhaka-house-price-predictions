import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.logger import logging
from src.house_price_prediction.utils.main_utils import save_object, read_yaml_file
from src.house_price_prediction.constants import SCHEMA_FILE_PATH
from src.house_price_prediction.config.config import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    def get_data_transformer_object(self):
        """
        Creates the preprocessing object for numerical and categorical columns.
        """
        try:
            # Fetching from schema.yaml to ensure consistency
            num_cols = self._schema_config.get("NUMERICAL_COLUMNS", [])
            cat_cols = self._schema_config.get("CATEGORICAL_COLUMNS", [])

            # Numerical Pipeline: Handle missing values + Scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline: Handle missing values + OneHot Encoding + Scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = self._schema_config.get("TARGET_COLUMN", {}).get("name", "price")

            # 1. Extract Target Features FIRST
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # 2. Extract Input Features (Removing the redundant 'axis=1' lines)
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            input_feature_test_df = test_df.drop(columns=[target_column_name])

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # 3. Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 4. Concatenate features with target
            input_feature_train_arr = np.array(input_feature_train_arr)
            input_feature_test_arr = np.array(input_feature_test_arr)

            # Using np.array() ensures the target is formatted correctly for concatenation
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 5. Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # 6. Save transformed data
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
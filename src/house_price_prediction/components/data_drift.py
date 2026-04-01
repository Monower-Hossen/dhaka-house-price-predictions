import os
import sys
import pandas as pd
from pandas import DataFrame

# Ensure evidently is installed: pip install evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.logger import logging
from src.house_price_prediction.utils.main_utils import write_yaml_file
from src.house_price_prediction.config.config import DataValidationConfig

class DataDriftDetector:
    def __init__(self, config: DataValidationConfig):
        try:
            self.data_validation_config = config
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detects if the distribution of data has changed between reference and current datasets.
        Saves the report to the configured path.
        """
        try:
            logging.info("Starting Data Drift detection using Evidently")
            
            ref_df = reference_df.copy().reset_index(drop=True)
            curr_df = current_df.copy().reset_index(drop=True)

            for col in ref_df.columns:
                # Evidently's internal Pydantic models can fail when validating specialized Pandas types (ExtensionArrays).
                # We force standard numpy-backed types: float for numeric, and object-dtype strings for categorical.
                if pd.api.types.is_numeric_dtype(ref_df[col]):
                    ref_df[col] = ref_df[col].astype(float)
                    curr_df[col] = curr_df[col].astype(float)
                else:
                    # astype(str).astype(object) ensures we don't use Pandas 2.0+ nullable string types
                    ref_df[col] = ref_df[col].astype(str).astype(object)
                    curr_df[col] = curr_df[col].astype(str).astype(object)

            # Initialize the report with DataDriftPreset
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=ref_df, current_data=curr_df)
            
            report_dict = drift_report.as_dict()

            # Save the drift report to artifacts
            report_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=report_path, content=report_dict)
            logging.info(f"Drift report saved at: {report_path}")

            # Extract drift status
            drift_status = False
            for metric in report_dict.get("metrics", []):
                if "dataset_drift" in metric.get("result", {}):
                    drift_status = metric["result"]["dataset_drift"]
                    break
            
            logging.info(f"Dataset drift detected: {drift_status}")
            return drift_status

        except Exception as e:
            raise CustomException(e, sys)

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    """Centralized paths for data ingestion."""
    root_dir: Path = Path("artifacts/data_ingestion")
    feature_store_file_path: Path = Path("artifacts/data_ingestion/data.csv")
    training_file_path: Path = Path("artifacts/data_ingestion/train.csv")
    testing_file_path: Path = Path("artifacts/data_ingestion/test.csv")
    source_data_path: str = "notebook/data/house_price.csv"

@dataclass
class DataTransformationConfig:
    """Paths for preprocessing objects."""
    root_dir: Path = Path("artifacts/data_transformation")
    preprocessor_obj_file_path: Path = Path("artifacts/data_transformation/preprocessor.pkl")
    transformed_train_file_path: Path = Path("artifacts/data_transformation/train.npy")
    transformed_test_file_path: Path = Path("artifacts/data_transformation/test.npy")

@dataclass
class DataValidationConfig:
    """Paths for data validation reports."""
    root_dir: Path = Path("artifacts/data_validation")
    drift_report_file_path: Path = Path("artifacts/data_validation/drift_report.yaml")

@dataclass
class ModelTrainerConfig:
    """Paths for trained models."""
    root_dir: Path = Path("artifacts/model_trainer")
    trained_model_file_path: Path = Path("artifacts/model_trainer/model.pkl")
    expected_accuracy: float = 0.6

# Schema Configuration
SCHEMA = {
    "columns": {
        "area": "int64",
        "bedrooms": "int64",
        "bathrooms": "int64",
        "stories": "int64",
        "mainroad": "object",
        "guestroom": "object",
        "basement": "object",
        "hotwaterheating": "object",
        "airconditioning": "object",
        "parking": "int64",
        "prefarea": "object",
        "furnishingstatus": "object"
    },
    "target_column": {"name": "price"}
}
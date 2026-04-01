import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Custom exception and logging would be imported here
from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.logger import logging
from src.house_price_prediction.utils.main_utils import read_sql_data
from src.house_price_prediction.config.config import DataIngestionConfig
from src.house_price_prediction.constants import DB_NAME

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        # logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Attempting to read data from MySQL database")
            df = read_sql_data()
            
            if df is None or df.empty:
                logging.warning("SQL data unavailable or empty, falling back to local CSV")
                if not os.path.exists(self.ingestion_config.source_data_path):
                    raise FileNotFoundError(f"Source data file not found at {self.ingestion_config.source_data_path}. "
                                            f"Please ensure the MySQL table '{DB_NAME}' exists or place the CSV file in the notebook/data directory.")
                df = pd.read_csv(self.ingestion_config.source_data_path)

            # Standardize column names to match schema.yaml (e.g., "No. Beds" -> "No_Beds")
            df.rename(columns={
                "No. Beds": "No_Beds",
                "No. Baths": "No_Baths",
                "Sub-region": "Sub_region"
            }, inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.training_file_path), exist_ok=True)

            df.to_csv(self.ingestion_config.feature_store_file_path, index=False, header=True)

            # logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.training_file_path,
                self.ingestion_config.testing_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
  
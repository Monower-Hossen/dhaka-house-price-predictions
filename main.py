import sys
from src.house_price_prediction.logger import logging
from src.house_price_prediction.exception import CustomException

# Components
from src.house_price_prediction.components.data_ingestion import DataIngestion
from src.house_price_prediction.components.data_validation import DataValidation
from src.house_price_prediction.components.data_transformation import DataTransformation
from src.house_price_prediction.components.model_trainer import ModelTrainer

# Configuration Manager
from src.house_price_prediction.entity.artifact_entity import DataIngestionArtifact
from src.house_price_prediction.config.configuration import ConfigurationManager

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # Initialize Configuration Manager
        config_manager = ConfigurationManager()

        # 1. Data Ingestion
        logging.info(">>>>>> Stage: Data Ingestion started <<<<<<")
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # 2. Data Validation (The New Gatekeeper)
        logging.info(">>>>>> Stage: Data Validation started <<<<<<")
        data_validation_config = config_manager.get_data_validation_config()
        
        # Construct artifact from ingestion results for validation
        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=train_data_path,
            test_file_path=test_data_path
        )

        data_validation = DataValidation(config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        
        data_validation_artifact = data_validation.initiate_data_validation()
        validation_status = data_validation_artifact.validation_status
        
        if not validation_status:
            logging.error(f"Data Validation Failed: {data_validation_artifact.message}")
            logging.error(f"Review the drift report for details: {data_validation_artifact.drift_report_file_path}")
            sys.exit(1) # Stop the pipeline execution safely
            
        logging.info("Data Validation Passed successfully")

        # 3. Data Transformation
        logging.info(">>>>>> Stage: Data Transformation started <<<<<<")
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # 4. Model Training
        logging.info(">>>>>> Stage: Model Training started <<<<<<")
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        r2, mae, rmse = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training complete. R2: {r2}, MAE: {mae}, RMSE: {rmse}")

        logging.info("Pipeline execution completed successfully")

    except Exception as e:
        raise CustomException(e, sys)
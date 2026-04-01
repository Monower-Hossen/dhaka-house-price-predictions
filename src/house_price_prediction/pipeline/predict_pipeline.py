import sys
import pandas as pd
import os
from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.utils.main_utils import load_object
from src.house_price_prediction.config.config import ModelTrainerConfig, DataTransformationConfig

class PredictPipeline:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
        self.transformation_config = DataTransformationConfig()

    def predict(self, features):
        try:
            model_path = self.model_config.trained_model_file_path
            preprocessor_path = self.transformation_config.preprocessor_obj_file_path

            # Load objects using the utility function
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform raw data and make prediction
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 Location: str,
                 Type: str,
                 No_Beds: int,
                 No_Baths: int,
                 Area: float,
                 Latitude: float,
                 Longitude: float,
                 Region: str,
                 Sub_region: str):
        
        # Mapping inputs to class attributes
        self.Location = Location
        self.Type = Type
        self.No_Beds = No_Beds
        self.No_Baths = No_Baths
        self.Area = Area
        self.Latitude = Latitude
        self.Longitude = Longitude
        self.Region = Region
        self.Sub_region = Sub_region

    def get_data_as_data_frame(self):
        try:
            # Keys must exactly match the column names used during model training
            custom_data_input_dict = {
                "Location": [self.Location],
                "Type": [self.Type],
                "No_Beds": [self.No_Beds],
                "No_Baths": [self.No_Baths],
                "Area": [self.Area],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude],
                "Region": [self.Region],
                "Sub_region": [self.Sub_region]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)
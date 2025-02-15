import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading model and preprocessor.")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            ##read the model and preprocessor from the file_path
            logging.info("Reading model and preprocessor from the file path.")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Debugging: Check if preprocessor is None
            if preprocessor is None:
                raise ValueError("Preprocessor not loaded correctly!")

            logging.info("Transforming features.")
            
            # Debugging: Print features before transformation
            print("Features before transformation:", features)
           
            logging.info("Transforming features.")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions.")
            prediction = model.predict(data_scaled)
            
            logging.info("Prediction completed.")
            return prediction
        except Exception as e:
            logging.info(f"Error in prediction: {e}")
            raise CustomException(e, sys)

##class that saves the input data and pass it to the next step wherever it is require and
##models will use this data to predict the output
class CustomData:
    def __init__(self,gender: str,race_ethnicity: str,parental_level_of_education: str,lunch: str,test_preparation_course: str,writing_score: float,reading_score: float):
                ##initialize/assigning all values into variables that are required for prediction
                self.gender = gender
                self.race_ethnicity = race_ethnicity  # Change to match expected column name
                self.parental_level_of_education = parental_level_of_education  # Change to match expected column name
                self.lunch = lunch
                self.test_preparation_course = test_preparation_course  # Change to match expected column name
                self.writing_score = writing_score  # Change to match expected column name
                self.reading_score = reading_score  # Change to match expected column name

    logging.info("Custom function for Converting data to DataFrame.")       
    def get_data_as_data_frame(self):
        ##return the data as dataframe
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],  # Change to match expected column name
                "parental level of education": [self.parental_level_of_education],  # Change to match expected column name
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],  # Change to match expected column name
                "writing score": [self.writing_score],  # Change to match expected column name
                "reading score": [self.reading_score]  # Change to match expected column name
            }
            logging.info("Converting custom data to DataFrame.")
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.error(f"Error in converting data to DataFrame: {e}")
            raise CustomException(e, sys)





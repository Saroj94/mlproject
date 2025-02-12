import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
##to test
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import Modeltrainer,ModelTrainerConfig

##class that saves the input data and pass it to the next step wherever it is require
##this decorator is used to create a class with attributes and methods, we can directly define the variable
@dataclass 
class DataIngestionConfig: ##it is just like providing the input thing that is required for dataingestion component
    ##data path where the data is stored
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','raw_data.csv')

##many functions then follow 
class DataIngestion:
    def __init__(self):
        ##initialize the class and the data ingestion_config store the data path
        self.ingestion_config = DataIngestionConfig()
    
    ##function that load the data from the path
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            ##read and load the data from the path(Can be from Api,database etc.)
            df = pd.read_csv('notebook/data/student.csv')
            logging.info("Read dataset as dataframe and Data loaded successfully")

            ##create folder artifact if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            ##split the data into train and test
            logging.info("Splitting data into train and test")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            
            logging.info("Data split into train and test for data ingestion is successfully completed") 

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Error while loading data from the path")
            raise CustomException("Error while loading data from the path", e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,tets_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)


    ##model training
    modeltrainer=Modeltrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,tets_arr))
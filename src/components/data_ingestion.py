from src.logger import logging
from src.exception import CustumException
import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

# Initialise Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion starts.....')
        try:
            df = pd.read_csv(os.path.join('notebook/data','gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe.....')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Train Test split.....')
            train_set, test_set = train_test_split(df,test_size=0.3,random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is completed.....')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info('Error occured at initiate data ingestion stage')
            raise CustumException(e,sys)
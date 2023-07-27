import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.logger import logging
from src.exception import CustumException
from src.utils import save_object
from src.utils import evaluate_model
import os
import sys
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Independent and Dependent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso': Lasso(),
                'ElasticNet':ElasticNet()
            }


            model_report : dict=evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('='*35)
            logging.info(f'Model Report : {model_report}')


            # To get Best Model score from dictionary
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model found ,  Model Name : {best_model_name} , R2_score : {best_model_score}")
            print('='*35)
            logging.info(f"Best Model found ,  Model Name : {best_model_name} , R2_score : {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )
                        

        except Exception as e:
            logging.info('Error occured at initiate model traing.....')
            raise CustumException(e,sys)
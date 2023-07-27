import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.exception import CustumException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info('Error occured at save object')
        raise CustumException(e,sys)
    

def evaluate_model(x_train,x_test,y_train,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # train model
            model.fit(x_train,y_train)

            # predict testing data
            y_pred = model.predict(x_test)


            test_model_score = r2_score(y_pred,y_test)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.info('Error occured at evaluate model training')
        raise CustumException(e,sys)
    



def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Error occured at load object')
        raise CustumException(e,sys)
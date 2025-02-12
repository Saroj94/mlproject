##Basic library
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
##modeling dependencies
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    ##create variable 
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class Modeltrainer:
    ##define the function
    def __init__(self):
        ##initializing the path name in the variable model_trainer_config
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test dataset")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "SVR": SVR(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boost regressor": GradientBoostingRegressor(),
                "xgboost": XGBRegressor(),
                "KNearest Neigbhour": KNeighborsRegressor(),
                "Decision Tree regressor": DecisionTreeRegressor()
            }

            model_report: dict = evaluate_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, model=models)

            ##to get the best model score
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best model score is {best_model_score}")

            ##get the best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            logging.info(f"Best model is {best_model_name}")
            if best_model_score < 0.6:
                logging.error("Model score is less than 0.6")
                raise CustomException("Best model score is less than 0.6", sys)

            ##preprocessing we can add if we want

            ##save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predict = best_model.predict(x_test)
            r2score = r2_score(y_test, predict)

            return r2score
        except Exception as e:
            raise CustomException(e, sys)
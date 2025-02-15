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

            ##hyperparameter tuning
            params={
                "Decision Tree regressor": {'criterion':['squared_error','friedman_mse','mean_absolute','poisson'],
                ##'splitter':['best','random'],
                ##'max_depth':[2,3,4,5,6,7,8,9,10]
            },
            "Random Forest Regressor":{'n_estimators':[8,16,32,64,100,200],
                                       ##'criterion':['mse','mae'],
                                       # maxfeatures:['auto','sqrt','log2'],
                                        ##'max_depth':[2,3,4,5,6,7,8,9,10]},
            },
            "Gradient Boost regressor":{'n_estimators':[8,16,32,64,128,256],
                                        ##'loss':['ls','lad','huber','quantile'],
                                        'learning_rate':[0.001,0.01,0.1,0.05],
                                        'subsample':[0.5,0.6,0.7,0.8,0.9,1.0],
                                        ##'criterion':['friedman_mse','mse','mae'],
                                        ##'max_depth':[2,3,4,5,6,7,8,9,10],
                                        ##'max_features':['auto','sqrt','log2'],
                                        ##'warm_start':[True,False],
                                        ##'validation_fraction':[0.1,0.2,0.3,0.4,0.5],
                                        ##'n_iter_no_change':[5,10,15,20],
                                        ##'tol':[0.001,0.01,0.1,0.2,0.3],
                                        ##'ccp_alpha':[0.0,0.1,0.2,0.3,0.4,0.5]},
            
            },
            "Linear Regression":{}, 
            "Lasso":{},
            "Ridge":{},
            "KNeighborsRegressor":{'n_neighbors':[5,7,9,11],
                                    ##'weights':['uniform','distance'],
                                    ##'algorithm':['auto','ball_tree','kd_tree','brute'],
                                    ##'leaf_size':[30,40,50,60],
                                    ##'p':[1,2]},
            },
            "SVR":{'kernel':['linear','poly','rbf','sigmoid'],
                    ##'degree':[2,3,4,5,6],
                    ##'gamma':['scale','auto'],
                    ##'C':[0.1,1,10,100,1000],
                    ##'epsilon':[0.1,0.2,0.3,0.4,0.5]},
            },
            "AdaBoost":{'n_estimators':[8,16,32,64,128,256],
                        'learning_rate':[0.1,0.01,0.001,0.05],
                        ##'loss':['linear','square','exponential']},
            },'XGBRegressor':{'n_estimators':[8,16,32,64,128,256],
                                'learning_rate':[0.1,0.01,0.001,0.05],
            }}

            model_report: dict = evaluate_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, 
                                                model=models,param=params)

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

"""##Basic library
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

            ##hyperparameter tuning
            params = {
                "Decision Tree regressor": {'criterion': ['squared_error', 'friedman_mse', 'mean_absolute', 'poisson']},
                "Random Forest Regressor": {'n_estimators': [8, 16, 32, 64, 100, 200]},
                "Gradient Boost regressor": {'n_estimators': [8, 16, 32, 64, 128, 256],
                                             'learning_rate': [0.001, 0.01, 0.1, 0.05],
                                             'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                "Linear Regression": {},
                "Lasso": {},
                "KNeighborsRegressor": {'n_neighbors': [5, 7, 9, 11]},
                "SVR": {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                "AdaBoost": {'n_estimators': [8, 16, 32, 64, 128, 256],
                             'learning_rate': [0.1, 0.01, 0.001, 0.05]},
                "XGBRegressor": {'n_estimators': [8, 16, 32, 64, 128, 256],
                                 'learning_rate': [0.1, 0.01, 0.001, 0.05]}
            }

            model_report: dict = evaluate_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, 
                                                models=models, param=params)

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
            raise CustomException(e, sys)"""
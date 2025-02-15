import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,
                   model,param):
    try:
        report={}

        for key, model_instance in model.items():
            para=param[key]
            
            gs=GridSearchCV(model,para,cv=3,n_jobs=-3)
            gs.fit(X_train,y_train) ##train model

            model_instance=gs.best_estimator_  ##best model
            model_instance.fit(X_train,y_train)

            ##prediction
            y_train_pred = model_instance.predict(X_train)  ##prediction on training set
            y_test_pred = model_instance.predict(X_test)  ##prediction on test set

            ##evaluate the model
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[key] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

"""def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    report = {}
    for model_name, model in models.items():
        logging.info(f"Training {model_name}")
        if model_name in param:
            gs = RandomizedSearchCV(model, param[model_name], cv=3, n_jobs=-1, verbose=2)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        report[model_name] = score
    return report"""

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        report={}

        for key, model_instance in model.items():
            model_instance.fit(X_train, y_train)  ##train model

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
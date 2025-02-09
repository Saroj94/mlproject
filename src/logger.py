import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m-%d-%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"log",LOG_FILE) ##how i want to store the log file
os.makedirs(log_path,exist_ok=True) ##create the log directory if it does not exist

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE) ##log file path

##configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
    )
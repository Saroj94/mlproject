import sys
import logging

##i customize the error message to show the error message and the line number where the error occured
def error_message_details(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message="error occured in python script name[{0}] at line number[{1}] error message[{2}]".format(
        filename,exc_tb.tb_lineno,str(error))
    
    return error_message

##custom exception class which is inherited from Exception class
class CustomException(Exception):
    """Base class for other exceptions"""
    def __init__(self, error_message, error_detail:sys):
        ##error message details populated from error_message_details function
        self.error_message = error_message_details(error_message,error_detail=error_detail)
        super().__init__(self.error_message)
    ##return the error message when we raise the exception
    def __str__(self):
        return self.error_message ##return the error message when we raise the exception
        
    
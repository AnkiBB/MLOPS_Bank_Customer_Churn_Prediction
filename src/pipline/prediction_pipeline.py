import sys
from src.entity.config_entity import ChurnPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class BankData:
    def __init__(self,
                credit_score,
                France,
                Germany,
                Spain,
                gender,
                age,
                tenure,
                balance,
                products_number,
                credit_card,
                active_member,
                estimated_salary
                ):
        """
        Bank Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.credit_score = credit_score
            self.France = France
            self.Germany = Germany
            self.Spain = Spain
            self.gender = gender
            self.age = age
            self.tenure = tenure
            self.balance = balance
            self.products_number = products_number
            self.credit_card = credit_card
            self.active_member = active_member
            self.estimated_salary = estimated_salary

        except Exception as e:
            raise MyException(e, sys) from e

    def get_bank_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from BankData class input
        """
        try:
            
            bank_input_dict = self.get_bank_data_as_dict()
            return DataFrame(bank_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_bank_data_as_dict(self):
        """
        This function returns a dictionary from BankData class input
        """
        logging.info("Entered get_bank_data_as_dict method as BankData class")

        try:
            input_data = {
                "credit_score": [self.credit_score],
                "France": [self.France],
                "Germany": [self.Germany],
                "Spain": [self.Spain],
                "gender": [self.gender],
                "age": [self.age],
                "tenure": [self.tenure],
                "balance": [self.balance],
                "products_number": [self.products_number],
                "credit_card": [self.credit_card],
                "active_member": [self.active_member],
                "estimated_salary": [self.estimated_salary]
            }

            logging.info("Created bank data dict")
            logging.info("Exited get_bank_data_as_dict method as bankData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class BankDataClassifier:
    def __init__(self,prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of BankDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of BankDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)
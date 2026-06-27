import pandas as pd
import numpy as np
import os
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path,output_path):
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        self.model = None

        self.X_train_scaled,self.X_test_resampled,self.y_train_resampled,self.y_test = None,None,None,None
        self.scaler = None
        self.smote = None

        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Model Training Started")

    def load_data(self):
        try:
            self.X_train_resampled = joblib.load(os.path.join(self.processed_data_path,"X_train_resampled.pkl"))
            self.X_test_scaled = joblib.load(os.path.join(self.processed_data_path,"X_test_scaled.pkl"))
            self.y_train_resampled = joblib.load(os.path.join(self.processed_data_path,"y_train_resampled.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path,"y_test.pkl"))

            logger.info("Data Loaded Successfully")

        except Exception as e:
            logger.error("Error While loading data")
            raise CustomException("Failed to Load Data",e)

    def train_model(self):
        try:
            self.model = LogisticRegression(random_state=42,max_iter=1000)
            self.model.fit(self.X_train_resampled,self.y_train_resampled)

            joblib.dump(self.model,os.path.join(self.output_path,"model.pkl"))
            logger.info("Model Training and Saving Done")
        
        except Exception as e:
            logger.error("Error Model Training and Saving")
            raise CustomException("Failed Model Training and Saving",e)
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test_scaled)

            accuracy = accuracy_score(self.y_test,y_pred)
            precision = precision_score(self.y_test,y_pred,average="weighted")
            f1 = f1_score(self.y_test,y_pred,average="weighted")
            recall = recall_score(self.y_test,y_pred,average="weighted")

            logger.info(f"Accuracy: {accuracy}, Precision: {precision}, F1: {f1}, Recall: {recall}")

        except Exception as e:
            logger.error("Error while model Evaluation")
            raise CustomException("Failed to Evaluate the Model",e)
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__=="__main__":
    model_trainer = ModelTraining(processed_data_path="artifacts/preprocessed",output_path="artifacts/model")
    model_trainer.run()
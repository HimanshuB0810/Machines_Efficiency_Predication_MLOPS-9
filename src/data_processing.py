import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class Data_Processing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.features = None
        
        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Data Processing Started")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data Loaded Successfully")
        except Exception as e:
            logger.error("Error While loading data")
            raise CustomException("Failed to Load Data",e)
        
    def preprocess(self):
        try:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'],errors="coerce")

            categorical_cols = ['Operation_Mode','Efficiency_Status']
            for col in categorical_cols:
                self.df[col] = self.df[col].astype(dtype="category")

            numerical_cols = self.df.select_dtypes(include=[np.number]).columns

            self.df['Year'] = self.df['Timestamp'].dt.year
            self.df['Month'] = self.df['Timestamp'].dt.month
            self.df['Day'] = self.df['Timestamp'].dt.day
            self.df['Hour'] = self.df['Timestamp'].dt.hour

            self.df.drop(columns=['Timestamp','Machine_ID'],inplace=True)

            label_encoder1 = LabelEncoder()
            self.df['Efficiency_Target'] = label_encoder1.fit_transform(self.df['Efficiency_Status'])
            label_encoder2 = LabelEncoder()
            self.df['Operation_Mode_Labels'] = label_encoder2.fit_transform(self.df['Operation_Mode'])
            
            logger.info("Basic Data Preprocessing Done")

        except Exception as e:
            logger.error("Error While Preprocessing data")
            raise CustomException("Failed to Preprocess Data",e)
        
    def split_scale_and_save(self):
        try:
            self.features=['Operation_Mode_Labels', 'Temperature_C', 'Vibration_Hz',
            'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
            'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
            'Predictive_Maintenance_Score', 'Error_Rate_%','Year', 'Month', 'Day', 'Hour'
                ]
            X = self.df[self.features]
            y = self.df['Efficiency_Target']    

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(random_state=42)   
            X_train_resampled,y_train_resampled = smote.fit_resample(x_train_scaled,y_train)
            
            joblib.dump(X_train_resampled,os.path.join(self.output_path,"X_train_resampled.pkl"))
            joblib.dump(X_test_scaled,os.path.join(self.output_path,"X_test_scaled.pkl"))
            joblib.dump(y_train_resampled,os.path.join(self.output_path,"y_train_resampled.pkl"))
            joblib.dump(y_test,os.path.join(self.output_path,"y_test.pkl"))

            joblib.dump(scaler,os.path.join(self.output_path,"scaler.pkl"))



            logger.info("Data Processing Done")

        except Exception as e:
            logger.error("Error While Split and scale data")
            raise CustomException("Failed to Split and scale Data",e)
        
    def run(self):
        self.load_data()
        self.preprocess()
        self.split_scale_and_save()

if __name__=="__main__":
    data_processor = Data_Processing(input_path="artifacts/raw/data.csv",output_path="artifacts/preprocessed")
    data_processor.run()

            
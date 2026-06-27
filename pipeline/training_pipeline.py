from src.model_training import ModelTraining
from src.data_processing import Data_Processing

if __name__=="__main__":
    data_processor = Data_Processing(input_path="artifacts/raw/data.csv",output_path="artifacts/preprocessed")
    data_processor.run()

    model_trainer = ModelTraining(processed_data_path="artifacts/preprocessed",output_path="artifacts/model")
    model_trainer.run()
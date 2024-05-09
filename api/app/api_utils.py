from joblib import load
from sklearn.linear_model import Ridge
from pydantic import BaseModel
from pandas import DataFrame
import os

def get_model() -> Ridge:
    model_path = os.environ.get('MODEL_PATH', 'model/ridge_model.pkl')
    model = load(model_path)
    return model

def transform_to_data(class_model: BaseModel) -> DataFrame:
    transition_dictionary = {key:[value] for key, value in class_model.dict().items()}
    data_frame = DataFrame(transition_dictionary)
    return data_frame

'''
with open(model_path, 'rb') as model_file:
        model = load(BytesIO(model_file.read()))
'''
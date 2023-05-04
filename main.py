from pathlib import Path
import numpy as np
from joblib import load

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.ml.data import process_data
from src.ml.model import inference
from config import path, cat_features_api

model_path = Path(path/'models')#
encoder = load(model_path.joinpath('encoder.joblib'))
model = load(model_path / 'random_forest.joblib')

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to a model prediction API!"}


def remove_hyphen(string: str) -> str:
    return string.replace('-', '_')


# define prediction base model
class InputData(BaseModel):

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(None, alias='education-num')
    marital_status: str = Field(None, alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(None, alias='capital-gain')
    capital_loss: int = Field(None, alias='capital-loss')
    hours_per_week: int = Field(None, alias='hours-per-week')
    native_country: str = Field(None, alias='native-country')

@app.post("/prediction/")
async def create_item(data: InputData):
    '''
    Function to take data and return a prediction
    :param data: the pydantic data model
    :return: prediction based on model and data provided
    '''
    data = dict(data)
    vals = np.array(list(data.values()))
    vals = vals.reshape((1, 14))
    df = pd.DataFrame(columns=data.keys(), data=vals)
    pred_data, _, _, _ = process_data(df, categorical_features=cat_features_api, training=False, encoder=encoder)
    prediction = inference(model, pred_data)
    prediction = int(prediction)
    return {'prediction:': prediction}

from pathlib import Path
from typing import Union

import numpy as np
from joblib import load

import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from src.ml.data import process_data
from src.ml.model import inference

# ingest data - define the response body with all the data features
# our perdict function then transforms the data (we need to load in the model and encode)
# and return a result

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

path = Path.cwd() / 'models'
encoder = load(path / 'encoder.joblib')
model = load(path / 'random_forest.joblib')

app = FastAPI()


@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to a model prediction API!"}


def remove_hyphen(string: str) -> str:
    return string.replace('-', '_')


# define prediction base model
class InputData(BaseModel):
    # age: int
    # workclass: object
    # fnlgt: int
    # education: object
    # education_num: Annotated[Union[int, None], Header()]
    # marital_status: Annotated[Union[str, None], Header()]
    # occupation: object
    # relationship: object
    # race: object
    # sex: object
    # capital_gain: Annotated[Union[int, None], Header()]
    # capital_loss: Annotated[Union[int, None], Header()]
    # hours_per_week: Annotated[Union[int, None], Header()]
    # native_country:  Annotated[Union[str, None], Header()]
    # salary: object

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

    # class Config:
    #     alias_generator = remove_hyphen


@app.post("/prediction/")
async def create_item(data: InputData):
    print(data)
    # return data

    data = dict(data)
    vals = np.array(list(data.values()))
    vals = vals.reshape((1, 14))
    df = pd.DataFrame(columns=data.keys(), data=vals)
    print(df)
    pred_data, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder)
    print(pred_data)
    prediction = inference(model, pred_data)
    print(prediction)
    prediction = int(prediction)
    return {'prediction:': prediction}

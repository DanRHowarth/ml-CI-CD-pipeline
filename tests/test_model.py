import pytest

from src.ml.model import train_model, compute_model_metrics, inference


# get data in

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_train_model():

    train_model(model, X_train, y_train)

    assert X_train.shape[0] == y_train.shape[0]

    assert columns are as expected


def test_compute_model_metrics():

    compute_model_metrics()


    assert

    assert



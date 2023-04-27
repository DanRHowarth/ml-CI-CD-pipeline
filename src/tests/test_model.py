from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml.data import process_data
from src.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier

# Import data and set up variables for testing
path = Path.cwd().parent.parent
data = pd.read_csv(path / 'data' / 'census.csv')

train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb,)

def test_train_model():
    rf_model = RandomForestClassifier()
    trained_model = train_model(rf_model, X_train, y_train)

    assert hasattr(trained_model, "classes_"), 'Model has not been trained'


def test_compute_model_metrics():
    model = RandomForestClassifier()
    trained_model = train_model(model, X_train, y_train)
    preds = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert 0 <= precision <= 1, 'Precision value are incorrect'
    assert 0 <= recall <= 1, 'Recall value are incorrect'
    assert 0 <= fbeta <= 1, 'Fbeta value are incorrect'

def test_inference():

    model = RandomForestClassifier()
    trained_model = train_model(model, X_train, y_train)
    preds = inference(trained_model, X_test)

    for pred in preds:
        assert pred == 0 or pred == 1, 'Model is predicting incorrect classes'

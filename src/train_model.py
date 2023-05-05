# Script to train machine learning model.
# from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import dump
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, \
    inference, assess_data_slices
from config import path, cat_features

# path = Path.cwd().parent
# cat_features = [
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
# ]

# load data and split

if __name__ == '__main__':
    data = pd.read_csv(path / 'data' / 'census.csv')
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb,
    )

    # Train and save a model.
    rf = RandomForestClassifier()

    trained_model = train_model(rf, X_train, y_train)

    preds = inference(trained_model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # save the model and encoder
    dump(trained_model, path / 'models' / 'random_forest.joblib')
    dump(encoder, path / 'models' / 'encoder.joblib')

    slice_data = assess_data_slices(test, encoder, lb, rf, cat_features)

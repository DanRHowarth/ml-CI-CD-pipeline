# Script to train machine learning model.
from pathlib import Path
import argparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import dump
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, \
    inference, assess_data_slices

path = Path.cwd().parent

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


def go(args):
    data = pd.read_csv(path / 'data' / 'census.csv')
    train, test = train_test_split(data, test_size=args.test_size)

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

    if args.save_artifacts:
        dump(trained_model, path / 'models' / 'random_forest.joblib')
        dump(encoder, path / 'models' / 'encoder.joblib')

    # assess performance on slices and save to data folder
    assess_data_slices(test, encoder, lb, rf, cat_features, save_slice=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model and encoder "
                                                 "against training dataset "
                                                 "and test on test set")

    parser.add_argument(
        "--test_size",
        type=float,
        help="Percentage of dataset for testing, as a decimal",
        default=0.2,
        required=False
    )

    parser.add_argument(
        "--save_artifacts",
        type=bool,
        help="Whether to save the trained model and encoder",
        default=False,
        required=False
    )

    args = parser.parse_args()

    go(args)

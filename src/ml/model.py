from pathlib import Path

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(model, X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    Model: scikit-learn model
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    return model.fit(X_train, y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def assess_data_slices(test: pd.DataFrame, encoder, lb, model,
                       cat_features: list,
                       save_slice: bool = True) -> pd.DataFrame:
    """
    Assess model performance of different slices of the test set and return a
    dataframe of results. Test set is required to be slices prior to processing
    and prediction. This function will go through all the unique values of
    the cat_features specified in the cat_features parameters and return scores
    for each of them.
    :param save_slice: if True, saves the slice as a txt file in data folder
    :param test:pd.DataFrame - test set, including data and labels
    :param encoder: pre-trained encoder for the data
    :param lb: pretrained label binarizer for the labels
    :param model: pre-trained ml model
    :param cat_features: list of categorical features to slice data on
    :return: pd.DataFrame of results
    """

    cat_list = []
    sub_cat_list = []
    precision_list = []
    recall_list = []
    fbeta_list = []

    for cat in cat_features:
        for sub_cat in test[cat].unique():
            # Proces the test data with the process_data function.
            X_test, y_test, _, _ = process_data(
                test, categorical_features=cat_features, label="salary",
                training=False, encoder=encoder, lb=lb,
            )

            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            cat_list.append(cat)
            sub_cat_list.append(sub_cat)
            precision_list.append(precision)
            recall_list.append(recall)
            fbeta_list.append(fbeta)

    slice_data = {'category': cat_list, 'sub_Category': sub_cat_list,
                  'precision': precision_list,
                  'recall': recall_list, 'fbeta': fbeta_list}

    slice_df = pd.DataFrame(slice_data)

    path = Path.cwd().parent / 'data'

    if save_slice:
        slice_df.to_csv(path / 'slice_data.txt', index=False, sep='\t')

    return slice_df

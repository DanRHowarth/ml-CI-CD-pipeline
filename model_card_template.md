# Model Card

## Model Details

## Intended Use

## Training Data

## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

## Caveats and Recommendations


Model Details
Justin C Smith created the model. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.

Intended Use
This model should be used to predict the acceptability of a car based off a handful of attributes. The users are prospective car buyers.

Metrics
The model was evaluated using F1 score. The value is 0.8960.

Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation). The target class was modified from four categories down to two: "unacc" and "acc", where "good" and "vgood" were mapped to "acc".

The original data set has 1728 rows, and a 75-25 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

Bias
According to Aequitas bias is present at the unsupervised and supervised level. This implies an unfairness in the underlying data and also unfairness in the model. From Aequitas summary plot we see bias is present in only some of the features and is not consistent across metrics.
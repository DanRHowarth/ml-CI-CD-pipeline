# Model Card

## Model Details
Dan Howarth created the model. It is a Random Forest Classifier using the default hyperparameters in scikit-learn 0.24.2. 

## Intended Use
This model should be used to predict a salary for an individual given socio-economic and job related data about that 
individual. The users for this model could be employment agencies, recruiters, HR departments looking to set salaries,
or social scientists undertaking research.

## Training Data
The data is 1994 Census data, avialbe online here - https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
The model was evaluated on an unseen test set of the original data. It was evaluated on the full dataset, and on slices
of the data to detect for bias. 

## Metrics
The model was assessed using Precision, Recall and Fbeta scores, and scores as follows:
- Precision: .7419
- Recall 0.6257 
- Fbeta: 0.6789

## Ethical Considerations
The data contains socio-economic information that needs to be handled sensitively. The model could be used to predict
salary information about individuals and this should be discouraged. 

## Caveats and Recommendations
THe data used to trained the model is from 1994 and is therefore nearly 30 years old. Please interpet any model results
accordingly.

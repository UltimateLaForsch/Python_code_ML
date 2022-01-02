import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Get data
iris = datasets.load_iris()
X = iris.data[:99]
y = iris.target[:99]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params,
                  nfold=3, num_boost_round=5,
                  metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params,
                  nfold=3, num_boost_round=5,
                  metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((cv_results["test-auc-mean"]).iloc[-1]))


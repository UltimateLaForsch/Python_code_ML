import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ames = pd.read_csv('Data/ames_housing_trimmed_processed.csv')
X = ames.iloc[:, :56]
y = ames.iloc[:, -1]
# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(X, y)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds,
                        metrics="rmse", as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "rmse"]))

# Early stopping
# Perform cross-validation with early stopping: cv_results
num_boosting_rounds = 100
cv_results = xgb.cv(nfold=3, seed=123, early_stopping_rounds=10, num_boost_round=num_boosting_rounds,
                    params=params, dtrain=housing_dmatrix, metrics='rmse', as_pandas=True)

# Print cv_results
if len(cv_results) < num_boosting_rounds:
    print('Early stop at round:', len(cv_results))
print(cv_results)

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

housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective": "reg:linear", "max_depth": 3}
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()

xgb.plot_importance(xg_reg)
plt.show()
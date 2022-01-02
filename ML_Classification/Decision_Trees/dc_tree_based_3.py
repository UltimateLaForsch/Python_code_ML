import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import math

dataset = pd.read_csv('data/auto.csv')
y = dataset.iloc[:, 0]
X = dataset.iloc[:, 1:7]
origin = X.iloc[:, 4]
encoder = LabelEncoder()
origin = encoder.fit_transform(origin).reshape(-1, 1)
one_hot = OneHotEncoder()
one_encoded = one_hot.fit_transform(origin)
X[one_hot.categories_[0]] = one_encoded.toarray()
X = X.drop(columns=['origin'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3)

# print(dataset)
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13,
                           random_state=3)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt = MSE(y_test, y_pred)
rmse_dt = math.sqrt(mse_dt)
print("Regression Tree Test set RMSE of dt: {:.2f}".format(rmse_dt))

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = MSE(y_test, y_pred_lr)
rmse_lr = math.sqrt(mse_lr)
print("Linear Regression Test set RMSE of dt: {:.2f}".format(rmse_lr))

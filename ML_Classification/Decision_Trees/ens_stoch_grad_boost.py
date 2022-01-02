import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

bike_sharing = pd.read_csv('data/bikes.csv')
y = bike_sharing['cnt']
del bike_sharing['cnt']
X = bike_sharing

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #, random_state=SEED)

sgbr = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2,
                                 subsample=0.9, max_features=0.75,)
sgbr.fit(X_train, y_train)
y_pred = sgbr.predict(X_test)
mse_test = MSE(y_test, y_pred)
rmse_test = mse_test ** 0.5
print('Test set RMSE of Stochastic Gradient Boosting: {:.3f}'.format(rmse_test))
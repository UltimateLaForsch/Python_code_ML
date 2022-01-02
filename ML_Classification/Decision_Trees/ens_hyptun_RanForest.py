import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

bike_sharing = pd.read_csv('data/bikes.csv')
y = bike_sharing['cnt']
del bike_sharing['cnt']
X = bike_sharing

SEED = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #, random_state=SEED)

rf_default = RandomForestRegressor(random_state=SEED)
rf_default.fit(X_train, y_train)
y_pred = rf_default.predict(X_test)
rmse_default = MSE(y_test, y_pred) ** 0.5
print('RMSE of Random Forest with default values: {:.3f}'.format(rmse_default))

# Grid search
rf = RandomForestRegressor(random_state=SEED)
print(rf.get_params())
params_rf = {'n_estimators': [100, 350, 500],
             'max_features': ['log2', 'auto', 'sqrt'],
             'min_samples_leaf': [2, 10, 30]}
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error',
                       cv=3, verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)
# Best Hyperparameters
best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)
# Best estimator
best_model = grid_rf.best_estimator_
y_pred_grid = best_model.predict(X_test)
rmse_grid = MSE(y_test, y_pred_grid) ** 0.5
print('Test RMSE of best model: {:.3f}'.format(rmse_grid))


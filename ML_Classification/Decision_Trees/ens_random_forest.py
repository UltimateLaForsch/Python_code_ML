import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('data/bikes.csv')
y = bike_sharing['cnt']
del bike_sharing['cnt']
X = bike_sharing

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

dt = DecisionTreeClassifier(random_state=2)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
rmse_test = MSE(y_test, y_pred) ** 0.5
print('Test set RMSE of a single Decision Tree: {:.2f}'.format(rmse_test))

rf = RandomForestRegressor(n_estimators=25, random_state=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred) ** 0.5
print('Test set RMSE of a Random Forest (ensemble): {:.2f}'.format(rmse_test))

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_, index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

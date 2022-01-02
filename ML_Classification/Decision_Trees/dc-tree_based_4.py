# Import train_test_split from sklearn.model_selection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as MSE

# Set SEED for reproducibility
SEED = 1

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

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26)

# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = -1 * cross_val_score(dt, X_train, y_train, cv=10,
                                scoring='neg_mean_squared_error', n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean()) ** 0.5

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = MSE(y_train, y_pred_train) ** 0.5

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))
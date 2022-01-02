import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

np.random.seed(42)
# Dataset Preparation
dataset = load_wine()
x = dataset.data
y = dataset.target

# print(f"target names: {dataset.target_names}")
# print(f"DESCR:\n{dataset.DESCR}")

df = pd.DataFrame(x, columns=dataset.feature_names)
df["y"] = y
df.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# CART Classifier: GridSearchCV
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 8, 10]
}
clf = DecisionTreeClassifier()
grid_cv = GridSearchCV(clf, parameters, cv=10, n_jobs=-1)
grid_cv.fit(x_train, y_train)

print(f"Parameters of best model: {grid_cv.best_params_}")
print(f"Score of best model: {grid_cv.best_score_}")

# Cart Classifier: Train Best Model
clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Decision Tree Accuracy: {round(score, 3)}")


# RandomForest Classifier: GridSearchCV
parameters = {
    'n_estimators': [10, 20, 40, 80, 160],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 8, 10]
}
clf_rf = RandomForestClassifier()
grid_cv_rf = GridSearchCV(clf_rf, parameters, cv=10, n_jobs=-1)
grid_cv_rf.fit(x_train, y_train)
print(f"Parameters of best model: {grid_cv_rf.best_params_}")
print(f"Score of best model: {grid_cv_rf.best_score_}")

# RandomForest Classifier: Train Best Model
clf_rf = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators=20)
clf_rf.fit(x_train, y_train)
score_rf = clf_rf.score(x_test, y_test)
print(f"Random Forest Accuracy: {round(score_rf, 3)}")

# GradientBoosting Classifier: Grid Search CV
# parameters = {
#     "loss": ["deviance", "exponential"],
#     'n_estimators': [10, 20, 40],
#     'criterion': ['mse', 'friedman_mse'],
#     'max_depth': [None, 2, 4, 8]
# }
# clf_gb = GradientBoostingClassifier()
# grid_cv_gb = GridSearchCV(clf_gb, parameters, cv=10, n_jobs=-1)
# grid_cv_gb.fit(x_train, y_train)
# print(f"Parameters of best model: {grid_cv.best_params_}")
# print(f"Score of best model: {grid_cv.best_score_}")

from sklearn.ensemble import GradientBoostingClassifier

# GradientBoosting Classifier: Train Best Model
clf_gb = GradientBoostingClassifier(criterion='friedman_mse', max_depth=2, n_estimators=20, loss="deviance")
clf_gb.fit(x_train, y_train)
score = clf_gb.score(x_test, y_test)
print(f"Gradient Boosting Accuracy: {round(score, 3)}")

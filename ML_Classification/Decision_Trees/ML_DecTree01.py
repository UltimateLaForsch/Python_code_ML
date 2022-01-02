import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from dtreeviz.trees import dtreeviz

np.random.seed(42)

dataset = load_wine()
X = dataset.data
y = dataset.target
type(dataset)

print(f"target names: {dataset.target_names}")
print(f"DESCR:\n{dataset.DESCR}")

df = pd.DataFrame(X, columns=dataset.feature_names)
df["y"] = y
df.head()

# CART Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 8, 10],
    'max_features':['auto', 'sqrt', 'log2'],
    'min_samples_leaf':[1, 2],
    'min_samples_split':[2, 4]
}
# Grid search - finding the optimal combination of hyperparameter values
clf = DecisionTreeClassifier()
grid_cv = GridSearchCV(clf, parameters, cv=10, n_jobs=-1)
grid_cv.fit(X_train, y_train)
print(f"Parameters of best model: {grid_cv.best_params_}")
print(f"Score of best model: {round(grid_cv.best_score_, 3)}")
# Train Best Model
clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Best model results:")
print(f"Accuracy: {round(score, 3)}")

viz = dtreeviz(clf, X, y,
               target_name="Wine Class",
               feature_names=dataset.feature_names,
               class_names=list(dataset.target_names),
               title="Decision Tree - Wine Dataset")
viz.save("decision_tree.svg")
viz.view()


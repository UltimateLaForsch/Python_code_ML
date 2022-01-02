from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = load_digits()
X = dataset.data
y_raw = dataset.target
# Transform into binary problem (digit '2' is one, all other digits are zero
transform_binary = lambda x: 1 if x == 2 else 0
y = np.array(list(map(transform_binary, y_raw)))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2) #3, random_state=1, stratify=y)

# We set random_state=0 for reproducibility
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
             'loss': ['hinge', 'log'], 'penalty': ['l1', 'l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
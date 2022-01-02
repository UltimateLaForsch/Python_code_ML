from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = load_digits()
X = dataset.data
y_raw = dataset.target
# Transform into binary problem (digit '2' is one, all other digits are zero
transform_binary = lambda x: 1 if x == 2 else 0
y = np.array(list(map(transform_binary, y_raw)))

# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)

# Report the best parameters
print("Best CV params", searcher.best_params_)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

indian_liver = pd.read_csv('data/indian_liver_patient_preprocessed.csv')
X = indian_liver.iloc[:, 1:11]
y = indian_liver['Liver_disease']

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # , random_state=SEED)
dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train, y_train)
y_pred_proba = dt.predict_proba(X_test)[:, 1]
dt_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Show Hyperparameters
print(dt.get_params())
# Set search value boundaries
params_dt = {'max_depth': [2, 3, 4],
             'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]}
# Perform Grid Search
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, cv=5, scoring='roc_auc', n_jobs=-1)
grid_dt.fit(X_train, y_train)
# Best Hyperparameters
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)
# Best estimator
best_model = grid_dt.best_estimator_
# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
# Compute test_roc_auc
grid_dt_roc_auc = roc_auc_score(y_test, y_pred_proba)
# Print roc_auc
print('Decision Tree with default Hyperparameters:\n Test set ROC AUC'
      ' score: {:.3f}'.format(dt_roc_auc))
print('Decision Tree with optimal Hyperparameters found by Gridsearch:\n'
      ' Test set ROC AUC score: {:.3f}'.format(grid_dt_roc_auc))
print('Best estimator: {}', best_model)

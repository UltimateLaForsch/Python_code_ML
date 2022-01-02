import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

ada.fit(X_train, y_train)
y_pred_proba = ada.predict_proba(X_test)[:, 1]
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
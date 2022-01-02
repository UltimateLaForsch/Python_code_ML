import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

indian_liver = pd.read_csv('data/indian_liver_patient_preprocessed.csv')
X = indian_liver.iloc[:, 1:11]
y = indian_liver['Liver_disease']

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, oob_score=True, random_state=1)

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
acc_oob = bc.oob_score_

print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

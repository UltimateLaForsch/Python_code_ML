import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


indian_liver = pd.read_csv('data/indian_liver_patient_preprocessed.csv')
X = indian_liver.iloc[:, 1:11]
y = indian_liver['Liver_disease']

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of a single decision tree: {:.2f}'.format(acc_test))

bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=SEED)

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bagging (50 trees): {:.2f}'.format(acc_test))

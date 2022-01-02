import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt


diabetes = pd.read_csv('../../diabetes.csv')
# print(diabetes.head)

X = diabetes.iloc[:, 0:7]
y = diabetes.iloc[:, 8]

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
plot_roc_curve(logreg, X_test, y_test)
plt.show()
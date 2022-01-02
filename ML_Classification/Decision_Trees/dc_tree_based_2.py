# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv('data/wbc.csv')
print(dataset)
X = dataset.iloc[:, 2:-1]
create_diagnosis = lambda x: 1 if(x == 'M') else 0
y = dataset['diagnosis'].map(create_diagnosis)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=8,
                                    random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

y_pred = dt_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred)

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=8,
                                    random_state=1)

# Fit dt_entropy to the training set
dt_gini.fit(X_train, y_train)

y_pred_gini = dt_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)


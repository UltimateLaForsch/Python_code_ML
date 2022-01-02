from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.load_digits()

# Description of the dataset
print(mnist['DESCR'])
# Keys of the dataset
print(mnist.keys())
print(mnist['feature_names'])
print(mnist['target'])
print(mnist.data.shape)
print(mnist.target.shape)
print(mnist.data)

# Create Feature matrix and Target vector
X = mnist.data
y = mnist.target

# Divide data in training and test samples
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Select and run model
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

# Make prediction with model results
y_pred = knn.predict(X_test)
print("'Test set predictions:")
print(y_pred)

# Calculate performance metric: Accuracy
print('Train Score: ', round(knn.score(X_train, y_train), 3))
print('Test Score: ', round(knn.score(X_test, y_test), 3))

# Display digit 1010
plt.imshow(mnist.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

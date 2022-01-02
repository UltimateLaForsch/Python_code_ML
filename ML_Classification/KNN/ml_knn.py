from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Get data
iris = datasets.load_iris()

# Create Features matrix and Target vector
X = iris.data
y = iris.target

# model selection
model = KNeighborsClassifier(n_neighbors=6)

# the model will be trained
model.fit(X, y)

# make a prediction for 2 new data sets with the trained model
test = np.array([[2.5, 5, 0.7, 1], [5.5, 3, 5, 2.5]])
print(model.predict(test))

from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(X, y, clf=model) #, test_idx=range(105, 150))
#
#
# import pandas as pd
# import pandas.plotting as pplot
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# df = pd.DataFrame(X, columns=iris.feature_names)
# print(df)
# plt.scatter(df['petal width (cm)'], df['petal length (cm)'], color='blue')
# plt.show()
# plt.scatter(df['sepal width (cm)'], df['sepal length (cm)'], color='green')
# plt.show()
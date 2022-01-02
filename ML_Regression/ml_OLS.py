import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import scatterplotmatrix

spy_set = pd.read_csv('spy.csv', header=None,
                      names=['Date', 'spy'])
spy_set['Date'] = spy_set['Date'].astype('datetime64[ns]')

X = (spy_set.index - spy_set.index[0]).values
X = X.reshape(-1, 1)
y = spy_set['spy'].to_numpy()
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(r2)

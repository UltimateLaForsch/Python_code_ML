import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

data = pd.read_csv('../../company-stock-movements-2010-2015-incl.csv')
movements = data.iloc[:, 1:]
companies = data.iloc[:, 0]

normalized_movements = normalize(movements)
model=TSNE(learning_rate=50)
tsne_features = model.fit_transform(normalized_movements, companies)
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]
plt.scatter(xs, ys, alpha=0.5)
# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

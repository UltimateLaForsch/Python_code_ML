import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dataset = pd.read_csv('../../seeds.csv', header=None)
samples = dataset.iloc[:, 0:7]
varieties = dataset.iloc[:, 7]
model = TSNE(learning_rate=200)
tsne_features = model.fit_transform(samples, varieties)
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]
plt.scatter(xs, ys, c=varieties)
plt.show()


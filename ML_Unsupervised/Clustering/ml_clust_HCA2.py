import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize

data = pd.read_csv('../../company-stock-movements-2010-2015-incl.csv')
movements = data.iloc[:, 1:]
companies = data.iloc[:, 0]
mov = movements.to_numpy()
com = companies.values.tolist()

normalized_movements = normalize(mov)
mergings = linkage(normalized_movements, method='complete')
dendrogram(mergings, labels=com, leaf_rotation=90, leaf_font_size=6)
plt.show()


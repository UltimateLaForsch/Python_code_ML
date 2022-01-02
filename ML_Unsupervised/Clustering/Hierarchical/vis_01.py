import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib import pyplot as plt
import seaborn as sns

comic_con = pd.read_csv('../Data/comic_com.csv')
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='ward',
                          metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2,
                                       criterion='maxclust')

colors = {1: 'red', 2: 'blue'}

# Plot a scatter plot
comic_con.plot.scatter(x='x_scaled', y='y_scaled',
                       c=comic_con['cluster_labels'].apply(lambda x: colors[x]))
plt.title('Vis. with Matplotlib')
plt.show()

sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.title('Vis. with Seaborn')
plt.show()

dn = dendrogram(distance_matrix)
plt.show()

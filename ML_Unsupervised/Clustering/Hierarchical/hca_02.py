import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns

comic_con = pd.read_csv('../Data/comic_com.csv')
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='ward',
                          metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2,
                                       criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.title('linkage - ward-method')
plt.show()

distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='single',
                          metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2,
                                       criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.title('linkage - single-method')
plt.show()

distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='complete',
                          metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2,
                                       criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.title('linkage - complete-method')

plt.show()
print()
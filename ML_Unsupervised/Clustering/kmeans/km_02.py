import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

comic_con = pd.read_csv('../Data/comic_com.csv')
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)
comic_con['cluster_labels'], distortion_list =\
    vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()

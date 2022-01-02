import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

dataset = pd.read_csv('../../seeds.csv', header=None)
samples = dataset.iloc[:, 0:7]
varieties = dataset.iloc[:, 7]
print(varieties)

# linage needs numpy arrays as input
samples = samples.to_numpy()
varieties = varieties.to_numpy()

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=4)
plt.show()

# Extension: Extract cluster labels for intermediate clustering
labels = fcluster(mergings, 6, criterion='distance')
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = load_iris()
unsuper_data = dataset.data
print(unsuper_data)
km = KMeans(n_clusters=3)
km.fit(unsuper_data)
labels = km.predict(unsuper_data)

xs = unsuper_data[:, 0]
ys = unsuper_data[:, 2]
# Assign the cluster centers: centroids
centroids = km.cluster_centers_
print(centroids)
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 2]

fig, ax = plt.subplots()
ax.scatter(xs, ys, c=labels, alpha=0.8)
ax.scatter(centroids_x, centroids_y, marker='D', s=80, c='orange')
# ax.legend()
plt.show()


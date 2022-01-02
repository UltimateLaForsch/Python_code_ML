import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

dataset = pd.read_csv('../../seeds.csv', header=None)
samples = dataset.iloc[:, 0:7]
varieties = dataset.iloc[:, 7]
# print(samples)
width = samples.iloc[:, 4]
length = samples.iloc[:, 3]
grains = pd.DataFrame()
grains['width'] = width
grains['length']= length
# print(grains)
model = PCA()
model.fit(grains)
mean = model.mean_
# Get 1st principal component
first_pc = model.components_[0, :]
fig, ax = plt.subplots()
ax.scatter(width, length)
ax.arrow(mean[0], mean[1], first_pc[0], first_pc[1],color='red', width=0.01)
ax.axis('equal')
ax.set_xlabel('Seed width')
ax.set_ylabel('Seed length')
plt.show()

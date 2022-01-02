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
# print(length)
fig, ax = plt.subplots()
ax.scatter(width, length)
ax.axis('equal')
ax.set_xlabel('Seed width')
ax.set_ylabel('Seed length')
plt.show()
# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)
# Display the correlation
print("Correlation: ", round(correlation, 3))
# Decorrelate
grains = pd.DataFrame()
grains['width'] = width
grains['length']= length
# print(grains)
model=PCA()
pca_features = model.fit_transform(grains)
xs = pca_features[:, 0]
ys = pca_features[:, 1]
fig, ax = plt.subplots()
ax.scatter(xs, ys)
ax.axis('equal')
ax.set_xlabel('Seed width')
ax.set_ylabel('Seed length')
plt.show()
correlation, pvalue = pearsonr(xs, ys)
print("Correlation after PCA: ", round(correlation, 3))
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

dataset = pd.read_csv('../../fish.csv', header=None)
df_samples = dataset.iloc[:, 1:7]
varieties = dataset.iloc[:, 0]
samples = df_samples.to_numpy()
print(samples)
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(samples)
features = range(pca.n_components_)
fig, ax = plt.subplots()
ax.bar(features, pca.explained_variance_)
ax.set_xlabel('PCA feature')
ax.set_ylabel('variance')
plt.show()


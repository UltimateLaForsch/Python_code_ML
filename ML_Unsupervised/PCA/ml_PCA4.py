import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale

dataset = pd.read_csv('../../fish.csv', header=None)
df_samples = dataset.iloc[:, 1:7]
varieties = dataset.iloc[:, 0]
samples = df_samples.to_numpy()
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)
# scaled_samples = scale(samples)
pca = PCA(n_components=2)
pca.fit(scaled_samples)
pca_features = pca.transform(scaled_samples)
print(pca_features.shape)


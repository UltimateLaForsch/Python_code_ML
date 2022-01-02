from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('../../fish.csv')
samples = data.iloc[:, 1:]
species = data.iloc[:, 0]
# print(samples)

scaler = StandardScaler()
kmeans = KMeans(n_clusters=4)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)
df = pd.DataFrame({'labels': labels, 'species': species})
# print(df)
ct = pd.crosstab(df['labels'], df['species'])
print(ct)


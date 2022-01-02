import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

dataset = pd.read_csv('../../Wikipedia articles/wikipedia-vectors.csv')
df_samples = dataset.iloc[:, 1:]
titles = dataset.columns.tolist()
titles.pop(0)
articles = df_samples.to_numpy()
articles = articles.transpose()
print(articles.shape)
print(type(articles))
# # print(type(articles))
svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)
pipeline.fit(articles)
labels = pipeline.predict(articles)
print(labels)
df = pd.DataFrame({'label': labels, 'article': titles})
print(df.sort_values('label'))


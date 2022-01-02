import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

dataset = pd.read_csv('../../Wikipedia articles/wikipedia-vectors.csv')
df_samples = dataset.iloc[:, 1:]
titles = dataset.columns.tolist()
titles.pop(0)
articles = df_samples.to_numpy()
articles = articles.transpose()
print(articles.shape)
print(type(articles))
titles = dataset.columns.tolist()
titles.pop(0)
dataset2 = pd.read_csv('../../Wikipedia articles/wikipedia-vocabulary-utf8.txt', header=None)
words = dataset2
model = NMF(n_components=6)
model.fit(articles)
nmf_features = model.transform(articles)
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
article = df.loc['Cristiano Ronaldo']
similarities = df.dot(article)
print(similarities.nlargest())

import pandas as pd
from sklearn.decomposition import NMF

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
# print(dataset2)
# words = dataset2.iloc[:, 0].tolist()
# print(words)
# print(type(words))

model = NMF(n_components=6)
model.fit(articles)
nmf_features = model.transform(articles)
# print(nmf_features.round(2))
df = pd.DataFrame(nmf_features, index=titles)
print(df)
print(df.loc['Anne Hathaway'])
print(df.loc['Denzel Washington'])

print(model.components_)
components_df = pd.DataFrame(model.components_, columns=words)
print(components_df)
print(components_df.shape)
# Select row 3: component
component = components_df.iloc[3, :]
# Print result of nlargest
# (gives the five words with the highest values for that component)
print(component.nlargest())

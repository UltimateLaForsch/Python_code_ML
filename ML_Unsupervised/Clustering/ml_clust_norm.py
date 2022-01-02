from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd

data = pd.read_csv('../../company-stock-movements-2010-2015-incl.csv')
movements = data.iloc[:, 1:]
companies = data.iloc[:, 0]
# print(movements)


normalizer = Normalizer()
kmeans = KMeans(n_clusters=10)
pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(movements)
labels=pipeline.predict(movements)
df = pd.DataFrame({'labels': labels, 'companies': companies})
print(df.sort_values(by='labels'))



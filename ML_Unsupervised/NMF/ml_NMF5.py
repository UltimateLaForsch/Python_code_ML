import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

dataset = pd.read_csv('../../Data/Musical artists/scrobbler-small-sample.csv')
dataset2 = pd.read_csv('../../Data/Musical artists/artists.csv', header=None)
artist_names = dataset2[0].tolist()
print(artist_names)
#print(type(artist_names))
ds = dataset.to_numpy()
artists = np.zeros((111, 500))
# print(ds)
for row in ds:
    artists[row[1], row[0]] = row[2]
scaler = MaxAbsScaler()
nmf = NMF(n_components=20)
normalizer = Normalizer()
pipeline = make_pipeline(scaler, nmf, normalizer)
norm_features = pipeline.fit_transform(artists)
df = pd.DataFrame(norm_features, index=artist_names)
print(df)
artist = df.loc['Bruce Springsteen', :]
print(artist)
similarities = df.dot(artist)
print(similarities.nlargest())

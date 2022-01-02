import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv('../../seeds.csv', header=None)
samples = dataset.iloc[:, 0:7]
varieties = dataset.iloc[:, 7]
print(varieties)

ks = range(1, 6)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(samples)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()

model = KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels
labels = model.fit_predict(samples)
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)
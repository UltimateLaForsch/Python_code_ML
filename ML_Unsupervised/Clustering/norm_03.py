import pandas as pd
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt

fifa = pd.read_csv('Data/fifa_18_sample_data.csv')
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])
fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter')
plt.show()
print(fifa[['scaled_wage', 'scaled_value']].describe())
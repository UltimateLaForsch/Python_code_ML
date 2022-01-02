# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:26:57 2021

@author: Ultimate LaForsch
"""

import pandas as pd
from scipy.cluster.vq import whiten, kmeans, vq
import matplotlib.pyplot as plt

fifa = pd.read_csv('Data/fifa_18_sample_data.csv')

fifa['sc_pac'] = whiten(fifa['pac'])
fifa['sc_dri'] = whiten(fifa['dri'])
fifa['sc_sho'] = whiten(fifa['sho'])
fifa['sc_eur_wage'] = whiten(fifa['eur_wage'])

scaled_features = ['sc_pac', 'sc_dri', 'sc_sho', 'sc_eur_wage']

cluster_centers, distortion = kmeans(fifa[['sc_pac', 'sc_dri', 'sc_sho']], 3)

fifa['cluster_labels'], distortion_list = vq(fifa[['sc_pac', 'sc_dri', 'sc_sho']], cluster_centers)

# Print the size of the clusters
print(fifa.groupby('cluster_labels')['ID'].count())

# Print the mean value of wages in each cluster
print(fifa.groupby('cluster_labels')['eur_wage'].mean())

fifa.groupby('cluster_labels')[scaled_features].mean().plot(kind='bar') 
plt.show() 

# Get the name column of first 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])
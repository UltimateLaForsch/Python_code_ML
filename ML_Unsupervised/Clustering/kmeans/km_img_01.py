# Import image class of matplotlib
import matplotlib.image as img
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import whiten
import seaborn as sns, matplotlib.pyplot as plt

r = []
g = []
b = []
# Read batman image and print dimensions
batman_image =img.imread('../Data/batman.jpg')
print(batman_image.shape)

# Store RGB values of all pixels in lists r, g and b
for row in batman_image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)
pixels = pd.DataFrame({'red': r,
                       'blue': b,
                       'green': g})
print(pixels.head())
batman_df_temp = pd.DataFrame(pixels)
batman_df = pd.DataFrame(whiten(batman_df_temp))
col_names = ['scaled_red', 'scaled_blue', 'scaled_green']
batman_df.columns = col_names
#batman_df['scaled_red'] = whiten(batman_df_temp['red'])

distortions = []
num_clusters = range(1, 7)
# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = \
        kmeans(batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], i)

    distortions.append(distortion)


# Create a data frame with two lists, num_clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters,
                           'distortions': distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()

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
                       'green': g,
                       'blue': b})
print(pixels.head())
batman_df_temp = pd.DataFrame(pixels)
batman_df = pd.DataFrame(whiten(batman_df_temp))
col_names = ['red', 'green', 'blue']
batman_df.columns = col_names

cluster_centers, distortion = \
        kmeans(batman_df[['red', 'green', 'blue']], 3)
colors = []
# Get standard deviations of each color
r_std, g_std, b_std = batman_df_temp[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    # Convert each standardized value to scaled value
    colors.append((
        scaled_r * r_std / 255,
        scaled_g * g_std / 255,
        scaled_b * b_std / 255
    ))

# Display colors of cluster centers
plt.imshow([colors])
plt.show()

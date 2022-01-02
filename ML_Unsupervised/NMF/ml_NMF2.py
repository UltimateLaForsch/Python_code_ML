import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF

dataset = pd.read_csv('../../lcd-digits.csv', header=None)
digits = dataset.to_numpy()
# Get row of first digit
digit = digits[0, :]
# print(digit)

# Reshape to bitmap
bitmap = digit.reshape(13, 8)
# The 1's in the bitmap draw the number 7 in digital style on the 0's canvas
print(bitmap)
# Draw bitmap of a digital 7
fig, ax = plt.subplots()
img1 = ax.imshow(bitmap, cmap='gray', interpolation='nearest')
fig.colorbar(img1)
plt.show()

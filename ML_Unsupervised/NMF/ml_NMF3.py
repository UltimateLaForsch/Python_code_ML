import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF


def show_as_image(sample):
    # Draw bitmap of a digital no.
    bitmap = sample.reshape(13, 8)
    fig, ax = plt.subplots()
    img1 = ax.imshow(bitmap, cmap='gray', interpolation='nearest')
    fig.colorbar(img1)
    plt.show()


dataset = pd.read_csv('../../lcd-digits.csv', header=None)
digits = dataset.to_numpy()

model = NMF(n_components=7)
# Apply fit_transform to samples: features
features = model.fit_transform(digits)
for component in model.components_:
    show_as_image(component)

digit_features = features[0, :]
print(digit_features)

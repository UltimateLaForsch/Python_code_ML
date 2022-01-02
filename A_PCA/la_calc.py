import numpy as np

x = np.array([1, 1, 1])
A = np.array([[1, 0, 0], [0, 2, -1], [0, -1, 3]])
y = np.array([2, -1, 0])
# y = x

norm = x.T.dot(A).dot(y)
print("Norm: Square root of ", norm)
x_diff = x - y
distance = x_diff.T.dot(A).dot(x_diff)
print("Distance: Square root of ", distance)
angles_nom = x.T.dot(A).dot(y)
angels_den = np.sqrt(x.T.dot(A).dot(x) * y.T.dot(A).dot(y))
# print(angles_nom, " # ", angels_den)
radians = np.arccos(angles_nom / angels_den)
degrees = np.rad2deg(radians)
print("Angle in Degrees between vectors: ", round(degrees,2))
print("Angle in Radians between vectors: ", round(radians, 2))




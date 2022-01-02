import numpy as np

x = np.array([1, 1, 1])
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
p = np.array([5/9, 10/9, 10/9])

x_diff = x - p
distance = x_diff.T.dot(A).dot(x_diff)
print("Distance: Square root of ", distance)
print("= ", np.sqrt(distance))

b1 = np.array([1, 1, 1])
b2 = np.array([0, 1, 2])
B = np.c_[b1, b2]
print(B)
p1 = B.T.dot(B)
print("p1: \n", p1)
p2 = np.linalg.inv(p1)
print("p2: \n", p2)
p3 = B.dot(p2)
print("p3: \n", p3)
p4 = p3.dot(B.T)
print("p4: \n", p4)
P = B.dot(np.linalg.inv(B.T.dot(B))).dot(B.T)
print("Projection Matrix: \n", P)
rank = np.linalg.matrix_rank(P)
print("Rank of Pro. Matrix: \n", rank)
x = np.array([12, 0, 0])
projection = P.dot(x)
print("Projection of x: \n", projection)


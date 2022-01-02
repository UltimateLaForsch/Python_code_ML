import numpy as np
m = np.array([[1, 7], [2, 4]])
cov = np.cov(m, ddof=0)
print(cov)
print(np.var(m, axis=1, ddof=0))

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

bc = load_breast_cancer()
X_pre = bc.data[:, 0:10]
scaler = StandardScaler()
X = scaler.fit_transform(X_pre)
y = bc.target
y[y == 0] = -1


def log_loss(raw_model_output):
    return np.log(1 + np.exp(-raw_model_output))


# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w @ X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s


# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
# C is set to disable Regularization
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X, y)
print(lr.coef_)

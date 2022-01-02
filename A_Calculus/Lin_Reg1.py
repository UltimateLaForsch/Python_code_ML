import numpy as np


# Here the function is defined
def linfit(xdat, ydat):
    # Here xbar and ybar are calculated
    xbar = np.sum(xdat) / len(xdat)
    ybar = np.sum(ydat) / len(ydat)

    # Insert calculation of m and c here. If nothing is here the data will be plotted with no linear fit
    m = np.sum((xdat - xbar) * ydat) / np.sum((xdat - xbar) ** 2)
    c = ybar - m * xbar
    # Return your values as [m, c]
    return [m, c]


# Scipy alternative
from scipy import stats

# Use the stats.linregress() method to evaluate regression
regression = stats.linregress(xdat, ydat)

line(regression)
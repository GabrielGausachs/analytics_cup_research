import numpy as np

def entropy(x):
    x = x[x > 0]
    return -(x * np.log(x)).sum()
# Koen Dercksen - 4215966
# Module containing impurity metrics.

from collections import Counter
import sys
import numpy as np
import time


def frequencies(data, target_attr):
    if len(data) == 0:
        return []
    try:
        c = Counter(data[:, target_attr])
        n_records = float(len(data))
        return np.array([v/n_records for v in c.values()])
    except TypeError:
        sys.stderr.write("Please use Numpy arrays!")
        sys.exit()


def gini(data, target_attr):
    freq = frequencies(data, target_attr)
    fs = np.square(freq)
    return 1 - np.sum(fs)

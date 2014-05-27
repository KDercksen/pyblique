# Koen Dercksen - 4215966
# Module containing impurity metrics.

from collections import Counter
from math import log
import sys


def frequencies(data, target_attr):
    try:
        c = Counter(data[:, target_attr])
        n_records = float(len(data))
        return {k: v/n_records for k, v in c.items()}
    except TypeError:
        sys.stderr.write("Please use Numpy arrays!")
        sys.exit()


def entropy(data, target_attr):
    freq = frequencies(data, target_attr)
    return -sum(f * log(f, 2) for f in freq.values())


def gini(data, target_attr):
    freq = frequencies(data, target_attr)
    return 1 - sum(f**2 for f in freq.values())


def classification_error(data, target_attr):
    freq = frequencies(data, target_attr)
    return 1 - max(freq.values())

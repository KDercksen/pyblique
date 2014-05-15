# Module containing impurity metrics.

from collections import defaultdict
from math import log


def frequencies(data, target_attr):
    freq = defaultdict(int)
    n_records = float(len(data))
    for record in data:
        freq[record[target_attr]] += 1/n_records
    return freq


def entropy(data, target_attr):
    freq = frequencies(data, target_attr)
    return -sum(f * log(f, 2) for f in freq.values())


def gini(data, target_attr):
    freq = frequencies(data, target_attr)
    return 1 - sum(f**2 for f in freq.values())


def classification_error(data, target_attr):
    freq = frequencies(data, target_attr)
    return 1 - max(freq.values())

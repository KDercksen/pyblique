# -*- coding: utf-8 -*-
# Koen Dercksen - 4215966

import impurity
import numpy as np
import sys


def get_data(fname):
    """Return a dictionary records/class-labels.
    Data should be accessed by their integer identifier.

        # Get the i-th record
        rec = data[i]

    Arguments:
    ----------
    fname: string
        The filename of the csv file to read data from. This file should
        contain float values exclusively!
    """
    try:
        data = np.genfromtxt(fname, comments="#", delimiter=",", dtype=float)
        return data
    except FileNotFoundError:
        sys.stderr.write("{} does not exist! Aborting.\n".format(fname))
        sys.exit(2)


class ObliqueClassifier:
    """Oblique classifier. Can be trained on a dataset and be used to
    predict unseen records.

    Currently, the classifier only accepts the gini index as a metric.
    """

    def __init__(self, metric=impurity.gini, data=None):
        """Metric can only be a minimizing function for now!
        """
        if data:
            self.train(data)
        self.metric = metric

    def train(self, data):
        pass

    def __possible_splits(self, data, attr):
        attr_vals = np.sort(data[:, attr])
        weights = np.repeat(1.0, 2) / 2
        return np.convolve(attr_vals, weights)[1:-1]

    def __best_ap_split(self, data, attr, metric):
        split_evals = {}
        for s in self.possible_splits(data, attr):
            cond = data[:, attr] <= s
            left, right = data[cond], data[~cond]
            split_evals[s] = metric(left, -1) + metric(right, -1)
        # Minimize because we're using gini index
        return min(split_evals, key=lambda x: split_evals[x])

    def __get_split_vector(self, data, metric):
        n_attrs = data.shape[1] - 1
        result = [self.best_ap_split(data, i, metric) for i in range(n_attrs)]
        return np.array(result)

    def __split_data(data, attr, threshold):
        cond = data[:, attr] <= threshold
        low, high = data[cond], data[~cond]
        return low, high

    def __is_leaf_node(self, data):
        labels = np.array(data[:, -1])
        return all(label == labels[0] for label in labels)

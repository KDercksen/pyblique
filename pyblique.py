# -*- coding: utf-8 -*-
# Koen Dercksen - 4215966

# from sklearn.tree import DecisionTreeClassifier as dtc
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
        self.tree = {}

    def fit(self, data):
        self.tree = self.__fit(data)

    def predict(self, record):
        cls = self.tree
        while type(cls) is dict:
            r_val = cls["index"]
            split = cls["split"]
            if record[r_val] <= split:
                cls = cls["low"]
            else:
                cls = cls["high"]
        return cls

    def __fit(self, data):
        isleaf, leaf = self.__is_leaf_node(data)
        if len(data) == 0:
            return -1
        elif isleaf:
            return leaf
        else:
            splitv = self.__get_split_vector(data)
            index, split = min(enumerate(splitv), key=lambda x: x[1][1])
            tree = {"index": index, "split": split[0]}
            low, high = self.__split_data(data, index, split[0])
            subtree_low = self.__fit(low)
            tree["low"] = subtree_low
            subtree_high = self.__fit(high)
            tree["high"] = subtree_high
        return tree

    def __best_split_on_attr(self, data, attr):
        # Will return a tuple of (split test, split value).
        attr_vals = np.sort(data[:, attr])
        weights = np.repeat(1.0, 2) / 2
        split_values = np.convolve(attr_vals, weights)[1:-1]
        split_evals = {}
        for s in split_values:
            cond = data[:, attr] <= s
            left, right = data[cond], data[~cond]
            split_evals[s] = self.metric(left, -1) + self.metric(right, -1)
        # Minimize because we're using gini index
        return min(split_evals.items(), key=lambda x: x[1])

    def __get_split_vector(self, data):
        n_attrs = data.shape[1] - 1
        result = [self.__best_split_on_attr(data, i) for i in range(n_attrs)]
        return np.array(result)

    def __split_data(self, data, attr, threshold):
        cond = data[:, attr] <= threshold
        low, high = data[cond], data[~cond]
        return low, high

    def __is_leaf_node(self, data):
        # Returns true/false and the class label (useful if this was a leaf)
        labels = np.array(data[:, -1])
        label_all = labels[0]
        return all(label == label_all for label in labels), label_all


# RUNNING SOME TESTS BREH
print("Loading data...")
data = get_data("Data/iris.data")

print("Initializing classifier...")
oc = ObliqueClassifier()

print("Fitting classifier...")
oc.fit(data)

print("Done training. Calculating error rate...")
labels = [oc.predict(r) for r in data]
actual_labels = data[:, -1]
total = len(labels)
incorrect = 0
for i, label in enumerate(labels):
    if label != actual_labels[i]:
        incorrect += 1
print("Error rate: {}".format(incorrect/total))

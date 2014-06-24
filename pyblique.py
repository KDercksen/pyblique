# -*- coding: utf-8 -*-
# Koen Dercksen - 4215966

import impurity
import numpy as np
from random import random
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

    Currently, the classifier only uses the gini index as a metric.
    """

    def __init__(self, metric=impurity.gini, data=None):
        """Metric can only be a minimizing function for now!
        """
        if data:
            self.train(data)
        self.metric = metric
        self.tree = {}

    def fit(self, data):
        self.tree = self.__create_decision_tree(data)

    def predict(self, record):
        cls = self.tree
        while type(cls) is dict:
            splitv = cls["split"]
            v = self.__checkrel(record, splitv) > 0
            if v:
                cls = cls["high"]
            else:
                cls = cls["low"]
        return cls

    def __create_decision_tree(self, data):
        if len(data) == 0:
            return -1
        isleaf, leaf = self.__is_leaf_node(data)
        if isleaf:
            return leaf
        else:
            splits = self.__get_all_splits(data)
            index, split = min(enumerate(splits), key=lambda x: x[1][1])
            # in order to make this oblique, we first have to build a vector
            # to enable the linear combination split
            splitv = np.zeros((len(data[0]),))
            splitv[-1] = -split[0]
            splitv[index] = 1
            # this doesnt quite work yet
            # splitv = self.__perturb(data, splitv, index)
            tree = {"split": splitv}
            low, high = self.__split_data(data, splitv)
            subtree_low = self.__create_decision_tree(low)
            tree["low"] = subtree_low
            subtree_high = self.__create_decision_tree(high)
            tree["high"] = subtree_high
        return tree

    def __checkrel(self, record, splitv):
        return sum(a*x for a, x in zip(record[:-1], splitv)) + splitv[-1]

    def __calc_u(self, record, splitv, attr):
        am = splitv[attr]
        top = am * record[attr] - self.__checkrel(record, splitv)
        return top / record[attr]

    def __perturb(self, data, splitv, attr):
        # WARNING SUPER HACKY METHOD
        # doesnt work yet i think
        p_repl = 0.3
        # calculate old impurity
        low, high = self.__split_data(data, splitv)
        old_imp = self.metric(low, -1) + self.metric(high, -1)
        us = np.sort(np.array([self.__calc_u(r, splitv, attr) for r in data]))
        weights = np.repeat(1.0, 2) / 2
        split_values = np.convolve(us, weights)[1:-1]
        splits = {}
        for s in split_values:
            cond = data[:, attr] <= s
            left, right = data[cond], data[~cond]
            splits[s] = self.metric(left, -1) + self.metric(right, -1)
        am = min(splits.items(), key=lambda x: x[1])
        new_splitv = np.array(splitv)
        new_splitv[attr] = am[0]
        # calculate new impurity
        low, high = self.__split_data(data, new_splitv)
        new_imp = self.metric(low, -1) + self.metric(high, -1)
        if old_imp > new_imp:
            return splitv
        elif old_imp == new_imp:
            if random() < p_repl:
                return new_splitv
            else:
                return splitv
        else:
            return new_splitv

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

    def __get_all_splits(self, data):
        n_attrs = data.shape[1] - 1
        result = [self.__best_split_on_attr(data, i) for i in range(n_attrs)]
        return np.array(result)

    def __split_data(self, data, splitv):
        high, low = [], []
        for i, record in enumerate(data):
            v = self.__checkrel(record, splitv) > 0
            if v:
                high.append(record)
            else:
                low.append(record)
        return np.array(low), np.array(high)

    def __is_leaf_node(self, data):
        # Returns true/false and the class label (useful if this was a leaf)
        labels = data[:, -1]
        label_all = labels[0]
        return all(label == label_all for label in labels), label_all


def cross_validate(data, n=5):
    # Return mean classification error, mean squared errors
    if len(data) < n:
        sys.stderr.write("Not enough data to perform {} splits!\n".format(n))
    print("Cross-validation with {} partitions.".format(n))
    splits = np.split(data, n)
    mse_errorsum = 0
    errorsum = 0
    for i in range(len(splits)):
        print("Iteration #{}".format(i + 1))
        tmpsplits = np.delete(splits, i, axis=0)
        print("Creating classifier tree...")
        oc = ObliqueClassifier()
        print("Fitting classifier...")
        oc.fit(np.concatenate(tuple(t for t in tmpsplits)))
        actual_labels = splits[i][:, -1]
        print("Predicting leftover records...")
        predictions = [oc.predict(r) for r in splits[i]]
        error = error_rate(predictions, actual_labels)
        print("Local error rate: {:.3f}".format(error))
        errorsum += error
        mse_errorsum += error ** 2
        print("Done.\n")
    return errorsum / n, mse_errorsum / n


def error_rate(predictions, labels):
    if len(predictions) != len(labels):
        sys.stderr.write("Incorrect array sizes ({} vs {}) please input evenly"
                         "sized arrays!".format(len(predictions), len(labels)))
    incorrect = 0
    for p, l in zip(predictions, labels):
        if p != l:
            incorrect += 1
    return incorrect/len(labels)


# RUNNING SOME TESTS BREH
data = get_data("Data/iris.data")
error, mse = cross_validate(data, n=10)
print("Error rate: {:.3f}\nMSE: {:.3f}".format(error, mse))

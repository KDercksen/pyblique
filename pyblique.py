# Koen Dercksen - 4215966
# import impurity
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
    """

    def __init__(self):
        pass

    def possible_splits(self, data, attr):
        """Get a list of possible split values for a certain attribute.
        For instance, image this:

            a = [[1], [2], [3], [4]]
            splits = possible_splits(a, 0)

        splits will be a generator that yields [1.5, 2.5, 3.5].

        Arguments:
        ----------
        data: array-like
            The dataset.
        attr: int
            The index of the attribute you want to calculate splits for.
        """
        attr_vals = np.sort(data[:, attr])
        weights = np.repeat(1.0, 2) / 2
        return np.convolve(attr_vals, weights)[1:-1]

    def best_ap_split(self, data, attr, metric):
        """Calculates the best (according to the metric) axis-parallel split
        on a certain attribute of the dataset.

        Arguments:
        ----------
        data: array-like
            The dataset.
        attr: int
            The attribute to calculate the best split for.
        metric: function
            Impurity measure (use the functions from the impurity module).
        """
        # This part should be moving to a higher level!
        split_evals = {}
        for s in self.possible_splits(data, attr):
            cond = data[:, attr] <= s
            left, right = data[cond], data[~cond]
            split_evals[s] = metric(left, -1) + metric(right, -1)
        # We need to minimize in case of gini index...
        # Probably need to maximize for other measures :(
        return min(split_evals, key=lambda x: split_evals[x])

    def get_split_vector(self, data, metric):
        """Return a list with the (locally) best splits per attribute.

        Arguments:
        ----------
        data: array-like
            The dataset.
        metric: function
            Impurity measure (use the function from the impurity module!)
        """
        n_attrs = data.shape[1] - 1
        result = [self.best_ap_split(data, i, metric) for i in range(n_attrs)]
        return np.array(result)

#!/usr/bin/env python

from pyblique import get_data, ObliqueClassifier, error_rate
from sklearn.cross_validation import KFold
import argparse
import os
import sys


class Tee:

    def __init__(self, *args):
        self.outputs = args

    def __call__(self, s="", end="\n"):
        for o in self.outputs:
            o.write(s + end)


def run(fname, folds):
    data = get_data("Data/{}.data".format(fname))
    with open("Results/{}_{}folds.txt".format(fname, folds), "w") as f:
        tee = Tee(sys.stdout, f)
        tee("Validating classifier with {}-fold test...".format(folds))
        kf = KFold(len(data), n_folds=folds)
        avg_error = 0
        it = 1
        for train, test in kf:
            tee("Iteration #{}".format(it))
            oc = ObliqueClassifier()
            oc.fit(data[train])
            predictions = [oc.predict(r) for r in data[test]]
            actual_labels = data[test][:, -1]
            error = error_rate(predictions, actual_labels)
            tee("Error: {:.3f}".format(error))
            tee()
            avg_error += error
            it += 1
        tee("Average error: {:.3f}".format(avg_error/folds))


if __name__ == "__main__":
    files = os.listdir("Data")
    files = [f.split(".")[0] for f in files]
    if not os.path.exists("Results"):
        os.makedirs("Results")
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folds", default=10, type=int,
                        help="Amount of folds")
    parser.add_argument("data", type=str, help="Name of dataset",
                        choices=files)

    args = parser.parse_args()
    run(args.data, args.folds)

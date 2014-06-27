#!/usr/bin/env python
# Code based on example code from scipy user guide.

from sklearn import datasets
import pylab as pl

iris = datasets.load_iris()
X = iris.data
Y = iris.target

pl.clf()

# Attributes: sepal length, sepal width, petal length, petal width
# 1 vs 2
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
pl.xlabel("Sepal length")
pl.ylabel("Sepal width")
pl.xticks(())
pl.yticks(())
pl.savefig("Report/images/sepal_length-sepal_width.png", bbox_inches="tight",
          transparent=True)
pl.clf()

# 1 vs 3
pl.scatter(X[:, 0], X[:, 2], c=Y, cmap=pl.cm.Paired)
pl.xlabel("Sepal length")
pl.ylabel("Petal length")
pl.xticks(())
pl.yticks(())
pl.savefig("Report/images/sepal_length-petal_length.png", bbox_inches="tight",
          transparent=True)
pl.clf()

# 1 vs 4
pl.scatter(X[:, 0], X[:, 3], c=Y, cmap=pl.cm.Paired)
pl.xlabel("Sepal length")
pl.ylabel("Petal width")
pl.xticks(())
pl.yticks(())
pl.savefig("Report/images/sepal_length-petal_width.png", bbox_inches="tight",
          transparent=True)
pl.clf()

# 2 vs 3
pl.scatter(X[:, 1], X[:, 2], c=Y, cmap=pl.cm.Paired)
pl.xlabel("Sepal width")
pl.ylabel("Petal length")
pl.xticks(())
pl.yticks(())
pl.savefig("Report/images/sepal_width-petal_length.png", bbox_inches="tight",
          transparent=True)
pl.clf()

# 2 vs 4
pl.scatter(X[:, 1], X[:, 3], c=Y, cmap=pl.cm.Paired)
pl.xlabel("Sepal width")
pl.ylabel("Petal width")
pl.xticks(())
pl.yticks(())
pl.savefig("Report/images/sepal_width-petal_width.png", bbox_inches="tight",
          transparent=True)
pl.clf()

# 3 vs 4
pl.scatter(X[:, 2], X[:, 3], c=Y, cmap=pl.cm.Paired)
pl.xlabel("Petal length")
pl.ylabel("Petal width")
pl.xticks(())
pl.yticks(())
pl.savefig("Report/images/petal_length-petal_width.png", bbox_inches="tight",
          transparent=True)
pl.clf()

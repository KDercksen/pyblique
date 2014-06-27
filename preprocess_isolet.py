#!/usr/bin/env python
# This little transformation makes the dataset a lot smaller
# but the classifier still takes absolute ages to fit it.

from pyblique import get_data
from sklearn.decomposition import PCA
import numpy as np


# Shrink the isolet dataset by means of PCA.
# 600+ attributes -> 10 attributes
isolet = get_data("Data/isolet.data")
pca = PCA(n_components=10)
transformed = pca.fit_transform(isolet[:, :-1])
labels = isolet[:, -1]
labels = np.reshape(labels, (labels.shape[0], 1))
result = np.concatenate((transformed, labels), axis=1)
np.savetxt("Data/isolet_compressed.data", result, delimiter=",")

from single import UncertaintySampling

import numpy as np

from skactiveml.classifier import SklearnClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def accuracy():
    X, y_true = make_classification(n_features=2, n_redundant=0, random_state=0)
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    us = UncertaintySampling(clf, X, y_true)
    us.entropy())

if __name__ == '__main__':
   accuracy() 

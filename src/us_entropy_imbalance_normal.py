from single import UncertaintySampling

import numpy as np

from skactiveml.classifier import SklearnClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def accuracy():
   X, y_true = make_classification(n_samples=1000)
   clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
   us = UncertaintySampling(clf, X, y_true)
   data = us.entropy(cycles=50)
   for i in range(50):
      print(data[i]["clf"].score(X, y_true))



if __name__ == '__main__':
   accuracy()

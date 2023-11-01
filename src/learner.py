import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from skactiveml.base import ClassFrequencyEstimator
from abc import ABC, abstractmethod

class Learner(ABC, BaseEstimator, ClassifierMixin):

   def __init__(self, clf: ClassFrequencyEstimator, y_true: np.ndarray) -> None:
      self.clf = copy.deepcopy(clf)
      self.y_true = y_true


   @abstractmethod
   def fit(self, X, y):
      pass

   @abstractmethod
   def predict(self, X):
      pass

   def score(self, X, y_expected):
      return self.clf.score(X, y_expected)

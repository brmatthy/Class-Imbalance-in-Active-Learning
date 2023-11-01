from single.al_trainer import al_single_step

from skactiveml.pool import UncertaintySampling as US

from src.learner import Learner


class UncertaintySampling(Learner):

    def __init__(self, clf, method="entropy") -> None:
        super.__init__(clf)
        self.qs = US(method=method)
     
    def fit(self, X, y):
        al_single_step(self.clf, self.qs, X, y, self.y_true)

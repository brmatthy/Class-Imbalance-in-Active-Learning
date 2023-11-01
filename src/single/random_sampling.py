from single.al_trainer import al_single_step

from skactiveml.pool import RandomSampling as RS 

from src.learner import Learner


class RandomSampling(Learner):

    def __init__(self, clf, random_state=0) -> None:
        super.__init__(clf)
        self.qs = RS(random_state=random_state)
     
    def fit(self, X, y):
        al_single_step(self.clf, self.qs, X, y, self.y_true)


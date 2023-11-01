from single.al_trainer import al_single

import numpy as np

from skactiveml.pool import UncertaintySampling as US
from skactiveml.utils import MISSING_LABEL


class UncertaintySampling:

    def __init__(self, clf, X, y_true, y=np.array([])) -> None:
        self.y = y
        if y.shape != y_true.shape:
            self.y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
        self.clf = clf;
        self.X = X;
        self.y_true = y_true;

    def entropy(self, cycles=10, output_cycles=None):
        return al_single(self.clf, US(method="entropy"), self.X, self.y, self.y_true, cycles=cycles, output_cycles=output_cycles)


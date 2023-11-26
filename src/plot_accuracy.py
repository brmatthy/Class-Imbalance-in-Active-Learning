from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(title: str, accuracies: [[List[float], str]], ax: plt.Axes):
    """ Plot the accuracies of the classification to a matplotlib pyplot.

        @param title (str) String for the title of the plot.
        @param accuracies (2-dimensional tuple, array like) First element of the tuple is a List of accuracies
            while the second is a string representing the title of the data. (ex: [[acc_1, "one"], [acc_2, "two"]])
        @param ax (matplotlib ax) the ax on which to plot the data.

        @returns nothing.
    """
    c = np.arange(len(accuracies[0][0]))
    for i in range(len(accuracies)):
        ax.plot(c, accuracies[i][0], label=accuracies[i][1])

    ax.set_ylabel('Balanced accuracy')
    ax.set_xlabel('Samples asked at oracle')
    ax.set_title(title)
    ax.legend()

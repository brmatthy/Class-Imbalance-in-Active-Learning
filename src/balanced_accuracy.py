from collections import Counter


def balanced_accuracy(y_true, y):
    """ Computes the balanced accuracy for a given prediction
    y_true and y must have the same dimensions.

    @param y_true (array like) List of the correct labels for the data set.
    @param y (array like) List of the predicted labels for the data set.

    @returns The weighted accuracy of the prediction for all the classes in the dataset.

    >>> balanced_accuracy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    1.0

    >>> balanced_accuracy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1],[0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    0.75

    >>> balanced_accuracy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    0.5
    """
    # The amount of samples from each class
    count = Counter()
    # The amount of correctly predicted samples from each class
    correct_count = Counter()
    for i in range(len(y_true)):
        true_label = y_true[i]
        count[true_label] += 1
        if true_label == y[i]:
            correct_count[true_label] += 1

    accuracy = 0
    # Predict the accuracy
    for key in dict(count):
        accuracy += correct_count[key]/count[key]

    accuracy /= len(dict(count))
    return accuracy

if __name__ == '__main__':
    import doctest
    doctest.testmod()
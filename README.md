# Class-Imbalance-in-Active-Learning

## Installation
The `requirements.txt` file [can be used](https://pip.pypa.io/en/stable/reference/requirements-file-format/) to install the dependencies.

## Project structure
For this project we used jupyter notebooks to format our project structure. 
The notebooks have a given order which can be used to see our thoughts throughout the research.

### Single
Firstly we got to know the sklearn Active Learning library and experimented with balanced and imbalanced data.

- [Accuracy](src/accuracy.ipynb): Compares the difference between accuracy on balanced and imbalanced datasets for active and not-active learning. 
- [Classifiers](src/classifiers.ipynb): Compares different classifiers to determine which ones are better suited for our purposes.
- [Query by committee](src/query_by_committee.ipynb): Implements the Query by Committee query strategy using the discovered classifiers.

### Batches
Next we took a look at batches to discover more realistic situations.

- [Greedy](src/greedy.ipynb): Implements the basic greedy batch policies and experiments with it.
- [Batch fairness](src/batch_fairness.ipynb): Compares the fairness of the different batch algorithms.

### Classes
We also experimented with the amount of classes to discover their impact.

- []
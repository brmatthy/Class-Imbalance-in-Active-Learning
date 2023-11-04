# Batch active learning

[Labeling a single sample per iteration](../single/accuract.ipynb) is a great way to train a model very cost efficient, even with imbalanced data. In practice however an annotator prefers to label a batch samples instead of a single sample.

Even with balanced data this already has downsides:
- Since all n samples are selected based on the same state of the model, they will not benefit from the info gained from the other samples in the same batch.
- The learner might pick samples that give more or less the same information. This may result in gaining very few new info, while having ladled a lot of samples.

There are 2 mayor ways of dealing with these problems.
- 1. [Greedy query strategy](./greedy.ipynb)
- 2. [Batch aware query strategy](./batch_aware.ipynb)

We will look into both approaches and how they deal with imbalanced data.
# Class Imbalance in Active Learning

## Introduction

Machine learning models often require large amounts of annotated data to train. This is
not a problem if open datasets such as Imagenet are available, but for most application
domains this will not be the case. A way to mitigate this problem is active learning:
trying to select the samples that will be most informative to the model, and then only
annotating those. This results in a feedback loop, where the model trains on a set of
samples, proposes new samples to label, which are labeled by an expert and added to
the dataset. This can be very useful for example if labeling requires specific expert
knowledge, such as a radiologist or biologist.

In this project, we investigate the role of class imbalance in the efficiency of Active
Learning. We hypothesize that class imbalance can have a great influence in the
effectiveness of Active Learning, as the majority class is likely to contain many
redundant samples which do not add much value to the learning algorithm, whereas the
minority class is more likely to contain more informative samples. This link between
class imbalance and Active Learning can be investigated using synthetic data or by
forcing different degrees of class imbalance in existing datasets, before moving on to
real-world imbalanced datasets.

## Literature and software

A quick overview of active learning can be found in [this blog](https://dsgissin.github.io/DiscriminativeActiveLearning/). For a more extensive
overview, see [this survey paper](https://burrsettles.com/pub/settles.activelearning.pdf). A very useful software library containing
implementations of active learning strategies is [scikit-activeml]x(https://github.com/scikit-activeml/scikit-activeml).

## Datasets
You will see that many experiments in active learning are done on simple, synthetic
data. This is because this allows you to vary important properties of the data, such as
imbalance and dimensionality, but also because active learning on very complex
datasets (such as images) is computationally much more expensive: hyperparameters for
neural networks must be found after each round of the feedback loop. A good overview
of why active learning is difficult on image data is given by Munjal et al. (2022).

We would recommend starting off with synthetic datasets, such as in the examples of
scikit-activeml. If this works well, you can move up to real-world tabular datasets, such
as the ones available in the [UCI repository](https://archive.ics.uci.edu) or on [OpenML](https://www.openml.org/). These repositories also
contain many imbalanced datasets on which you can verify your results on synthetic
data. Some examples are given in Dang et al. (2013) and Phoungphol et al. (2012). If you
are ambitious, you can also try out active learning techniques on smaller image datasets
like MNIST or CIFAR-10, but for this you will most likely need the HPC cluster. We would
recommend keeping this as an “extra”.

## References
- Dang, X. T., Hirose, O., Saethang, T., Tran, V. A., Nguyen, L. A. T., Le, T. K. T., ... & Satou, K.
(2013). A novel over-sampling method and its application to miRNA prediction. Journal of
Biomedical Science and Engineering, 6(02), 236.
- Munjal, P., Hayat, N., Hayat, M., Sourati, J., & Khan, S. (2022). Towards robust and
reproducible active learning using neural networks. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (pp. 223-232).
- Phoungphol, P., Zhang, Y., & Zhao, Y. (2012). Robust multiclass classification for learning
from imbalanced biomedical data. Tsinghua Science and technology, 17(6), 619-628.



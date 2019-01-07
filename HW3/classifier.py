from hw3_utils import *
import numpy as np

from euclidean_distance import euclidean_distance


class knn_factory(abstract_classifier_factory):

    def __init__(self, k=1):
        self._k = k

    def train(self, data, labels):
        """"
        train a classifier
        :param data: 2D numpy.ndarray of features
        :param labels: 1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :return: abstract_classifier object
        """
        return knn_classifier(data, labels, self._k)


class knn_classifier(abstract_classifier):

    def __init__(self, train_features, train_labels, k=1):
        """"
        construct a knn classifier
        :parameter
        :train_features: 2D numpy.ndarray of features
        :train_labels:   1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :k:              integer, determine the size of k-nearest-neighbors (subgroup of train_features).
                         equals 1 as default (1-nearest-neighbor).
        """

        self._train_features = train_features
        self._train_labels   = train_labels
        self._k              = k

    def _procedure_edited_NN(self):
        """
        reduce train samples space without damaging the classifier quality
        """
        pass # later if we won't be too tired

    def classify(self, features):
        """
        classify a new set of features
        :param features: 1D numpy.ndarray of features
        :return: a tagging of the given features (1 or 0)
        """

        # iterate through train data set and obtain the euclidean
        # distances between the sample we want to classify and every sample in the train data set.
        distances = np.asarray([euclidean_distance(features,
                                                   train_feature)
                                for train_feature
                                in self._train_features.tolist()])

        # partition the distances s.t k'th element in his final location if we were sorting the array
        # (all elements before it are smaller or equal)
        # idx contains the indexes of the re-ordered array
        idx = np.argpartition(distances, self._k)

        # get indexes of first k smallest distances (related to k nearest nearest neighbors)
        k_minimal_distances_idx    = idx[:self._k]

        # get labels of k nearest neighbors
        k_minimal_distances_labels = self._train_labels[k_minimal_distances_idx]

        # get the majority vote
        count_label_zero = 0
        count_label_one  = 1
        for label in k_minimal_distances_labels:
            if label == 0:
                count_label_zero += 1
            if label == 1:
                count_label_one += 1

        if count_label_zero > count_label_one:
            return 0
        else:
            return 1





from hw3_utils            import *
import numpy              as np
from sklearn              import tree
from sklearn.linear_model import Perceptron
from sklearn.ensemble     import AdaBoostClassifier
from evaluate             import evaluate
from sklearn.linear_model import LogisticRegression


from euclidean_distance import euclidean_distance
from evaluate           import *

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
        count_label_one = 0
        for label in k_minimal_distances_labels:
            if label == 0:
                count_label_zero += 1
            if label == 1:
                count_label_one += 1

        if count_label_zero > count_label_one:
            return 0
        else:
            return 1


class ID3_factory(abstract_classifier_factory):
    def __init__(self):
        pass

    def train(self, data, labels):
        """"
        train ID3 decision tree classifier
        :param data: 2D numpy.ndarray of features
        :param labels: 1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :return: abstract_classifier object
        """
        return ID3_classifier(data, labels)


class ID3_classifier(abstract_classifier):
    def __init__(self, train_features, train_labels):
        """"
        construct a ID3 decision tree classifier
        :parameter
        :train_features: 2D numpy.ndarray of features
        :train_labels:   1D numpy.ndarray of {1,0} labels (ordered as train_features)
        """

        self._train_features = train_features
        self._train_labels = train_labels
        self._tree = tree.DecisionTreeClassifier()
        self._tree = self._tree.fit(train_features, train_labels)

    def classify(self, features):
        """
        classify a new set of features
        :param features: 1D numpy.ndarray of features
        :return: a tagging of the given features (1 or 0)
        """
        # features vector needs to be reshaped to 1,-1 shape
        return self._tree.predict(features.reshape(1, -1))


class perceptron_factory(abstract_classifier_factory):
    def __init__(self):
        pass

    def train(self, train_features, train_labels):
        """"
        train perceptron classifier
        :param data: 2D numpy.ndarray of features
        :param labels: 1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :return: abstract_classifier object
        """
        return perceptron_classifier(train_features, train_labels)


class perceptron_classifier(abstract_classifier):
    def __init__(self, train_features, train_labels):
        """"
        construct a perceptron classifier
        :parameter
        :train_features: 2D numpy.ndarray of features
        :train_labels:   1D numpy.ndarray of {1,0} labels (ordered as train_features)
        """

        self._train_features = train_features
        self._train_labels = train_labels
        self._clf = Perceptron()
        self._clf.fit(train_features, train_labels)

    def classify(self, features):
        """
        classify a new set of features
        :param features: 1D numpy.ndarray of features
        :return: a tagging of the given features (1 or 0)
        """

        # features vector needs to be reshaped to 1,-1 shape
        return self._clf.predict(features.reshape(1, -1))


class logistic_regression_factory(abstract_classifier_factory):
    def __init__(self):
        pass

    def train(self, train_features, train_labels):
        """"
        train log regression classifier
        :param data: 2D numpy.ndarray of features
        :param labels: 1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :return: abstract_classifier object
        """
        return logistic_regression_classifier(train_features, train_labels)


class logistic_regression_classifier(abstract_classifier):
    def __init__(self, train_features, train_labels):
        """"
        construct a log regression classifier
        :parameter
        :train_features: 2D numpy.ndarray of features
        :train_labels:   1D numpy.ndarray of {1,0} labels (ordered as train_features)
        """

        self._train_features = train_features
        self._train_labels = train_labels
        self._LogRegressor = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
        self._LogRegressor.fit(train_features, train_labels)

    def classify(self, features):
        """
        classify a new set of features
        :param features: 1D numpy.ndarray of features
        :return: a tagging of the given features (1 or 0)
        """

        # features vector needs to be reshaped to 1,-1 shape
        return self._LogRegressor.predict(features.reshape(1, -1))


class adaboost_factory(abstract_classifier_factory):
    def __init__(self):
        pass

    def train(self, train_features, train_labels):
        """"
        train adaboost classifier
        :param data: 2D numpy.ndarray of features
        :param labels: 1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :return: abstract_classifier object
        """
        return adaboost_classifier(train_features, train_labels)


class adaboost_classifier(abstract_classifier):
    def __init__(self, train_features, train_labels):
        """"
        construct a log regression classifier
        :parameter
        :train_features: 2D numpy.ndarray of features
        :train_labels:   1D numpy.ndarray of {1,0} labels (ordered as train_features)
        """

        self._train_features = train_features
        self._train_labels = train_labels
        self._ada  = AdaBoostClassifier()
        self._ada.fit(train_features, train_labels)

    def classify(self, features):
        """
        classify a new set of features
        :param features: 1D numpy.ndarray of features
        :return: a tagging of the given features (1 or 0)
        """

        # features vector needs to be reshaped to 1,-1 shape
        return self._ada.predict(features.reshape(1, -1))


class majority_votes_factory(abstract_classifier_factory):
    def __init__(self):
        pass

    def train(self, train_features, train_labels):
        """"
        train adaboost classifier
        :param data: 2D numpy.ndarray of features
        :param labels: 1D numpy.ndarray of {1,0} labels (ordered as train_features)
        :return: abstract_classifier object
        """
        return majority_votes_classifier(train_features, train_labels)


class majority_votes_classifier(abstract_classifier):
    def __init__(self, train_features, train_labels):
        """"
        construct a log regression classifier
        :parameter
        :train_features: 2D numpy.ndarray of features
        :train_labels:   1D numpy.ndarray of {1,0} labels (ordered as train_features)
        """

        self._train_features = train_features
        self._train_labels = train_labels
        self._ada  = adaboost_classifier(train_features, train_labels)
        self._log  = logistic_regression_classifier(train_features, train_labels)
        self._id3  = ID3_classifier(train_features, train_labels)
        self._knn1 = knn_classifier(train_features,train_labels,k=1)
        self._knn3 = knn_classifier(train_features, train_labels, k=3)

    def classify(self, features):
        """
        classify a new set of features
        :param features: 1D numpy.ndarray of features
        :return: a tagging of the given features (1 or 0)
        """

        ada_prediction = self._ada.classify(features)
        log_prediction = self._log.classify(features)
        id3_prediction = self._id3.classify(features)
        knn1_prediction = self._knn1.classify(features)
        knn3_prediction = self._knn3.classify(features)

        OnesCount = 0
        for p in [ada_prediction, log_prediction, id3_prediction, knn1_prediction, knn3_prediction]:
            if p:
                OnesCount += 1

        if OnesCount > 2:
            return 1
        else:
            return 0





if __name__ == '__main__':
    perceptron_accuracy, perceptron_error = evaluate(perceptron_factory(), 2)
    logistic_accuracy, logistic_error = evaluate(logistic_regression_factory(), 2)
    adaboost_accuracy, adaboost_error     = evaluate(adaboost_factory(), 2)
    majority_accuracy, majority_error     = evaluate(majority_votes_factory(),2)

    print(perceptron_accuracy, perceptron_error)
    print(logistic_accuracy, logistic_error)
    print(adaboost_accuracy, adaboost_error)
    print(majority_accuracy, majority_error)






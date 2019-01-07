import sys
import numpy as np

from hw3_utils import *
from euclidean_distance import euclidean_distance
from classifier import knn_classifier

def main():
    try:
        train_features, train_labels, test_features = load_data()
    except Exception as e:
        raise ValueError("Error load data")

    try:
        train_features = np.asarray(train_features)
        train_labels   = np.asarray(train_labels)
        test_features  = np.asarray(test_features)
    except Exception as e:
        raise ValueError("Error convert data to numpy arrays")

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("Number of train samples and tags are not the same for god sake!")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Number of features in the train set and test set are not equal."
                         "they might be originated from different data sets for god sake!")

    knn_5 = knn_classifier(train_features, train_labels, k=5)








if __name__ == '__main__':

    sys.exit(main())

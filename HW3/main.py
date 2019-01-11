import sys
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from hw3_utils import *
from euclidean_distance import euclidean_distance
from split_crosscheck_groups import split_crosscheck_groups
from split_crosscheck_groups import load_k_fold_data
from evaluate import evaluate

from classifier import *

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

    # Standard Score Normalization
    try:
        ampirical_mean = np.mean(train_features, axis=0)
        ampirical_std  = np.std(train_features, axis=0)

        train_features = (train_features - ampirical_mean) / ampirical_std
        ######################## MANOR 11/1/19 ########################
        # After run the algorithm with and without normalization,     #
        # it seems that the result stay the same.                     #
        ###############################################################
    except Exception as e:
        raise ValueError('Error with normalization')


    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("Number of train samples and tags are not the same for god sake!")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Number of features in the train set and test set are not equal."
                         "they might be originated from different data sets for god sake!")

    try:
        split_crosscheck_groups(train_features, train_labels, num_folds=2)
    except Exception as e:
        raise e

    Ks        = [1, 3, 5, 7, 9, 11, 13, 15]
    Accuracys = []
    Errors    = []
    try:
        for k in Ks:
            knn_factory_kth = knn_factory(k)
            accuracy, error = evaluate(knn_factory_kth, 2)
            Accuracys.append(accuracy)
            Errors.append(error)
    except Exception as e:
        raise e

    results = [[k, acc, err] for k, acc, err in zip(Ks, Accuracys, Errors)]

    with open(os.path.join(os.getcwd(), 'kFold_results', 'experiments6.csv'), 'w') as exp6:
        wr = csv.writer(exp6)
        for res in results:
            wr.writerow(res)

    fig = plt.figure(figsize=(10,10))
    plt.plot(Ks, Accuracys, linestyle='--', marker='o', color='r', label='Accuracy, KNN')
    plt.plot(Ks, Errors, linestyle='--', marker='o', color='b', label='Error, KNN')
    for kth, acc, err in zip(Ks, Accuracys, Errors):
        plt.text(kth, acc, '{},{:.3f}'.format(kth, acc))
        plt.text(kth, err, '{},{:.3f}'.format(kth, err))
    plt.legend()
    plt.title('Accuracy and Error as Function of KNN Value')
    plt.savefig(os.path.join(os.path.join(os.getcwd(), 'kFold_results', 'experiments6')))





    ################################################# MANOR 11/1/19 #################################################
    # It seems that the best accuracy achieved with k=3.                                                            #
    # according to the following paper, perhaps better results might achieved using non-euclidean distance measure. #
    # https: // www.ncbi.nlm.nih.gov / pmc / articles / PMC4978658 /                                                #
    #################################################################################################################

    try:
            decision_tree_factory = ID3_factory()
            accuracy, error = evaluate(decision_tree_factory, 2)
    except Exception as e:
        raise e

    with open(os.path.join(os.getcwd(), 'kFold_results', 'experiments12.csv'), 'w') as exp12:
        wr = csv.writer(exp12)
        wr.writerow([1, accuracy, error])

    plt.plot(range(Ks[-1]), [accuracy]*Ks[-1], linestyle='--', marker='o', color='g', label='Accuracy, ID3')
    plt.plot(range(Ks[-1]), [error]*Ks[-1], linestyle='--', marker='o', color='y', label='Error, ID3')
    plt.text(Ks[-1], accuracy, '{},{:.3f}'.format(Ks[-1], accuracy))
    plt.text(Ks[-1], error, '{},{:.3f}'.format(Ks[-1], error))
    plt.legend()
    plt.title('Accuracy and Error as Function of KNN Value and Different Classifiers')
    plt.savefig(os.path.join(os.path.join(os.getcwd(), 'kFold_results', 'experiments12')))
    plt.show()















if __name__ == '__main__':

    sys.exit(main())

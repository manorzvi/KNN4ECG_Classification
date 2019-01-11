import os
import numpy as np
import re


def k_fold_train_and_test(k):
    """"
    assign the kth fold as test dataset and merge the rest into train dataset
    :parameter
    :k:                     number for the kth fold to be our test dataset
    :returns
    :train_features_array:  2D numpy.ndarray to hold all the train samples features. for example: 900x187
    :train_labels_array:    1D numpy.ndarray to hold all train labels. for example: 900x1
    test_features:          2D numpy.ndarray to hold all the test samples features. for example: 100x187
    :test_labels:           1D numpy.ndarray to hold all test labels. for example: 100x1
    """

    testData = object
    trainData = []

    cwd = os.getcwd()
    os.chdir('kFold_results')

    for filename in os.listdir():
        if not filename.endswith('.npz'):
            continue
        else:
            regex = r'ecg_fold_{}.data.npz'.format(k)
            if re.match(regex, filename):
                testData = np.load(filename)
            else:
                trainData.append(np.load(filename))

    test_features = testData['name1']
    test_labels   = testData['name2']

    train_features = []
    train_labels   = []
    for td in trainData:
        train_features.append(td['name1'])
        train_labels.append(td['name2'])

    train_features_array = train_features.pop(0)
    train_labels_array   = train_labels.pop(0)

    for tf, tl in zip(train_features, train_labels):
        train_features_array = np.concatenate((train_features_array, tf))
        train_labels_array   = np.concatenate((train_labels_array, tl))

    os.chdir(cwd)

    return train_features_array, train_labels_array,\
           test_features, test_labels


def evaluate(classifier_factory, k):
    """"
    the function performs k-fold cross validation(with k as number of folds) on the folded dataset,
    using classifier_factory instance provided.
    :parameter
    :classifier_factory: classifier_factory object
    :k                 : number of folds
    :returns
    :accuracy: (True Positive + True Negative) / N samples
    :error:    (False Positive + False Negative) / N samples
    """
    accuracy = 0
    error = 0
    i = 1
    for kfold in range(k):
        train_features, train_labels, test_features, test_labels = k_fold_train_and_test(kfold)

        knn_classifier = classifier_factory.train(train_features, train_labels)
        true_positive_count = 0
        true_negative_count = 0
        false_positive_count = 0
        false_negative_count = 0

        for test_feature, test_label in zip(test_features, test_labels):
            prediction = knn_classifier.classify(test_feature)
            if prediction == 1 and test_label == 1: # True Positive
                true_positive_count += 1
            if prediction == 1 and test_label == 0: # False Positive
                false_positive_count += 1
            if prediction == 0 and test_label == 1: # False Negative
                false_negative_count += 1
            if prediction == 0 and test_label == 0: # True Negative
                true_negative_count += 1

        current_accuracy = (true_positive_count + true_negative_count) / len(test_labels)
        current_error    = (false_positive_count + false_negative_count) / len(test_labels)

        # Nice trick to calculate average incrementally:
        # Mn = Mn-1 + (An -Mn-1)/n
        accuracy = accuracy + (current_accuracy - accuracy) / i
        error    = error    + (current_error-error)         / i

        i += 1
    return accuracy, error


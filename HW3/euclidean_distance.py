import numpy as np
def euclidean_distance(subjectOne_features, subjectTwo_features):
    """"
    Return the euclidean distance netween subjects
    :parameter
    :subjectOne_features: 1D numpy.ndarray of features
    :subjectTwo_features: 1D numpy.ndarray of features

    :returns
    :euclidean distance between subjects
    """
    return np.linalg.norm(subjectOne_features-subjectTwo_features)

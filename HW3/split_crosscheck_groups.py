import numpy as np
import os
import shutil

# We Could use here the obvious method: StratifiedKFold of sklearn.model_selection
# But it is forbidden. BuuuuZZZ


def create_result_directory(dir_name):
    result_directory_fullpath = os.path.join(os.getcwd(), dir_name)
    if not os.path.exists(result_directory_fullpath):
        os.makedirs(result_directory_fullpath)
        print(result_directory_fullpath + ' created')
    else:  # directory exist
        shutil.rmtree(result_directory_fullpath)
        print(result_directory_fullpath + ' removed')
        os.makedirs(result_directory_fullpath)
        print(result_directory_fullpath + ' created again')


def split_crosscheck_groups(features, labels, num_folds):
    """"
    get dataset and split it into <num_folds> sub-datasets, while keeps the ratio between '1' and '0' labels.
    save all in result directory under current location.
    :parameter
    :features:  2D numpy.ndarray - 1000 samples on 187 features
    :labels:    1D numpy.ndarray - 1000 labels
    :num_folds: integer          - number fo sub-datasets
    :returns
    :None
    """

    copy_features = np.copy(features)
    copy_labels   = np.copy(labels)
    num_features   = copy_features.shape[0]
    k_fold_size = num_features/num_folds

    # create results directory
    try:
        create_result_directory('kFold_results')
    except Exception as e:
        raise ValueError("Create result directory failed")

    # indexes of the samples with label '0'
    zero_labels_idx = np.where(copy_labels == 0)[0]
    # indexes of the samples with label '1'
    one_labels_idx = np.where(copy_labels == 1)[0]

    # shuffle arrays for randomness
    np.random.shuffle(zero_labels_idx)
    np.random.shuffle(one_labels_idx)

    # samples with label '0'
    zero_labeled_features = copy_features[zero_labels_idx]
    zero_labels           = copy_labels[zero_labels_idx]
    # samples with label '1'
    one_labeled_features  = copy_features[one_labels_idx]
    one_labels            = copy_labels[one_labels_idx]

    if len(zero_labels_idx) + len(one_labels_idx) != num_features:
        raise ValueError("Houston we have a problem.\nIt might caused by more then two classes labels group")

    # calculate ratio between labels count
    if len(zero_labels_idx) > len(one_labels_idx):
        ratio = len(zero_labels_idx) / len(one_labels_idx)
        how_many_ones  = int(k_fold_size / (1 + int(ratio)))
        how_many_zeros = int(k_fold_size - how_many_ones)
    else:
        ratio = len(one_labels_idx)/len(zero_labels_idx)
        how_many_zeros = int(k_fold_size/(1+int(ratio)))
        how_many_ones  = int(k_fold_size - how_many_zeros)

    # for each k_fold sub-dataset:
    # ----------------------------
    #   1. build features matrix according to the ratio
    #   2. build labels vector according to the ratio
    #   3. save to result directory
    #   4. pop those samples from arrays to avoid duplications
    for k in range(num_folds):
        # 1.
        k_fold_features = np.concatenate((one_labeled_features[:how_many_ones],
                                         zero_labeled_features[:how_many_zeros]))
        # 2.
        k_fold_labels   = np.concatenate((one_labels[:how_many_ones],
                                          zero_labels[:how_many_zeros]))
        k_fold_filename = os.path.join(os.getcwd(), 'kFold_results', 'ecg_fold_' + str(k) + '.data')

        if k == num_folds-1:
            # 1.
            k_fold_features = np.concatenate((one_labeled_features,
                                              zero_labeled_features))
            # 2.
            k_fold_labels = np.concatenate((one_labels,
                                            zero_labels))
            k_fold_filename = os.path.join(os.getcwd(), 'kFold_results', 'ecg_fold_' + str(k) + '.data')


        # 3.
        try:
            np.savez(k_fold_filename,
                 name1=k_fold_features,
                 name2=k_fold_labels)
            print(k_fold_filename + ' created')
        except Exception as e:
            raise ValueError("Create " + k_fold_filename + " failed")

        # 4.
        one_labeled_features  = one_labeled_features[how_many_ones:]
        zero_labeled_features = zero_labeled_features[how_many_zeros:]
        one_labels            = one_labels[how_many_ones:]
        zero_labels           = zero_labels[how_many_zeros:]


def load_k_fold_data(index):
    """"
    load <index> sub-dataset into tuple.
    :parameter
    :index:  integer - index of sun-dataset

    :returns
    :tuple: (features<index>, labels<index>)
    """
    cwd = os.getcwd()
    os.chdir('kFold_results')

    k_fold_filename = 'ecg_fold_' + str(index) + '.data.npz'
    data = np.load(k_fold_filename)
    os.chdir(cwd)
    return data['name1'], data['name2']




















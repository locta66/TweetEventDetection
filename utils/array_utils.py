import numpy as np
from sklearn import metrics
from scipy import sparse
from sklearn import preprocessing

import utils.multiprocess_utils as mu


def merge_array(array):
    res_list = list()
    for arr in array:
        res_list.extend(arr)
    return res_list


def random_array_items(array, item_num, keep_order=True):
    """ Get random items from array.
    :param array: From which items are extracted.
    :param item_num: Number of items to be extracted.
    :param keep_order: If remains the relative order within array in the result.
    :return:
    """
    array_len = len(array)
    item_num = array_len if item_num >= array_len else item_num
    item_index = np.random.choice(array_len, item_num, replace=False)
    item_index = sorted(item_index) if keep_order else item_index
    return [array[i] for i in item_index]


def array_partition(array, partition_arr=(1, 1, 1), random=True, ordered=False):
    indexes = index_partition(array, partition_arr, random)
    return [[array[j] for j in indexes[i]] for i in range(len(indexes))]


def index_partition(array, partition_arr=(1, 1, 1), random=True, ordered=False):
    item_num = len(array)
    indexes = np.array([i for i in range(item_num)])
    indexes = shuffle(indexes) if random else indexes
    normed_portion = np.array(partition_arr) / np.sum(partition_arr)
    sum_idx = [0]
    sum_normed_por = [0]
    for portion in normed_portion:
        sum_normed_por.append(sum_normed_por[-1] + portion)
        sum_idx.append(int(sum_normed_por[-1] * item_num))
    sum_idx[-1] = item_num
    return [indexes[sum_idx[i]: sum_idx[i + 1]] for i in range(len(partition_arr))]


def shuffle(array, inplace=True):
    array = array if inplace else array[:]
    np.random.shuffle(array)
    return array


def score(labels_true, labels_pred, score_type):
    if score_type == 'auc':
        return metrics.roc_auc_score(labels_true, labels_pred)
    elif score_type == 'nmi':
        return metrics.normalized_mutual_info_score(labels_true, labels_pred)
    elif score_type == 'homo':
        return metrics.homogeneity_score(labels_true, labels_pred)
    elif score_type == 'cmplt':
        return metrics.completeness_score(labels_true, labels_pred)


def group_array_by_condition(array, item_key):
    dictionary = dict()
    for item in array:
        item_key = item_key(item)
        if item_key not in dictionary:
            dictionary[item_key] = [item]
        else:
            dictionary[item_key].append(item)
    return [dictionary[key] for key in sorted(dictionary.keys())]


def sample_index(array):
    return np.random.choice(a=[i for i in range(len(array))], p=np.array(array) / np.sum(array))


def choice(array):
    return np.random.choice(array)


# def scale(x):
#     return preprocessing.scale(x)


# def cosine_similarity(x, y):
#     return metrics.pairwise.cosine_similarity(x, y)
#
#
# def cosine_matrix_single(contract_mtx, x_y_pairs):
#     matrix = contract_mtx.todense()
#     for idx, (x, y) in enumerate(x_y_pairs):
#         vecx = matrix[x].reshape([1, -1])
#         vecy = matrix[y].reshape([1, -1])
#         x_y_pairs[idx].append(cosine_similarity(vecx, vecy)[0][0])
#     return x_y_pairs


def cosine_similarity(vecarr1, vecarr2=None):
    return metrics.pairwise.cosine_similarity(vecarr1, vecarr2)


def cohesion_score(vecarr):
    vecnum = len(vecarr)
    if vecnum <= 1:
        return None
    cos_sim_mtx = cosine_similarity(vecarr)
    cohesion_score = paircnt = 0
    for i in range(0, vecnum - 1):
        for j in range(i + 1, vecnum):
            cohesion_score += cos_sim_mtx[i, j]
            paircnt += 1
    assert paircnt == vecnum * (vecnum-1) / 2
    return cohesion_score / paircnt


# if __name__ == '__main__':
#     matrix = [
#         [1, 0, 0],
#         [3, 1, 0],
#         [0, 0, 2],
#         [0, 4, 0],
#         [0, 8, 1],
#         [5, 4, 3],
#     ]
#     cosine_similarity(matrix)

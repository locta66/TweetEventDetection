import numpy as np
from sklearn import metrics
from scipy import sparse

import utils.multiprocess_utils as mu


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


def cosine_similarity(x, y):
    return metrics.pairwise.cosine_similarity(x, y)


def cosine_matrix_single(contract_mtx, x_y_pairs):
    matrix = contract_mtx.todense()
    for idx, (x, y) in enumerate(x_y_pairs):
        vecx = matrix[x].reshape([1, -1])
        vecy = matrix[y].reshape([1, -1])
        x_y_pairs[idx].append(cosine_similarity(vecx, vecy)[0][0])
    return x_y_pairs


def cosine_matrix_multi(vec_matrix, process_num=8):
    vec_num = len(vec_matrix)
    x_y_pairs = [[i, j] for i in range(1, vec_num - 1) for j in range(i + 1, vec_num)]
    pairs_block = array_partition(x_y_pairs, [1] * process_num, random=False)
    contract_mtx = sparse.csr_matrix(vec_matrix)
    res_list = mu.multi_process(cosine_matrix_single, [(contract_mtx, pairs) for pairs in pairs_block])
    res_list = merge_list(res_list)
    cosine_matrix = np.zeros([vec_num, vec_num])
    for x, y, cos in res_list:
        cosine_matrix[x][y] = cosine_matrix[y][x] = cos
    return cosine_matrix


if __name__ == '__main__':
    matrix = [
        [1, 0, 0],
        [3, 1, 0],
        [0, 0, 2],
        [0, 4, 0],
        [0, 8, 1],
        [5, 4, 3],
    ]
    cosine_matrix_multi(matrix, process_num=2)
    # print(cosine_similarity(np.array([1, 2, 5, 4]).reshape((1, -1)), [[1, 2, 3, 4.1], [5, 1.2, 5, 1.2]]))


METHOD_EXTEND = 'extend'
METHOD_APPEND = 'append'
_SUPPORTED_MERGE_METHODS = {METHOD_EXTEND, METHOD_APPEND}


def merge_list(array, method=METHOD_EXTEND):
    if method not in _SUPPORTED_MERGE_METHODS:
        raise ValueError('param method incorrect: {}'.format(method))
    res = list()
    for item in array:
        if method == METHOD_EXTEND:
            res.extend(item)
        elif method == METHOD_APPEND:
            res.append(item)
    return res
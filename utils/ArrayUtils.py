import numpy as np
from sklearn import metrics


def random_array_items(array, item_num, keep_order=True):
    """
    Get random items from array.
    :param array: From which items are extracted.
    :param item_num: Number of items to be extracted.
    :param keep_order: If remains the relative order within array in the result.
    :return:
    """
    array_len = len(array)
    item_num = array_len if item_num >= array_len else item_num
    indexes = np.random.choice(array_len, item_num, replace=False)
    indexes = sorted(indexes) if keep_order else indexes
    return [array[i] for i in indexes]


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
    sum_idx.append(item_num)
    return [indexes[sum_idx[i]: sum_idx[i + 1]] for i in range(len(partition_arr))]


def shuffle(array, new_arr=False):
    array[9]
    array = array[:] if new_arr else array
    np.random.shuffle(array)
    return array


# def roc(score_label_pairs, curve_points=150):
#     scoreidx = 0
#     score_label_pairs = sorted(score_label_pairs, key=lambda item: item[scoreidx], reverse=True)
#     max_threshold = score_label_pairs[0][scoreidx]
#     min_threshold = score_label_pairs[-1][scoreidx]
#     interval = (max_threshold - min_threshold) / curve_points
#     threshold_list = [min_threshold + i * interval for i in range(1, curve_points - 1)]
#
#     thresholds_per_process = fI.split_multi_format(threshold_list, 15)
#     param_list = [(score_label_pairs, process_threshold) for process_threshold in thresholds_per_process]
#     roc_curve_arr = fI.multi_process(roc_through_thresholds, param_list)
#     roc_curve = fI.merge_list(roc_curve_arr)
#     roc_curve.insert(0, (0.0, 0.0))
#     roc_curve.append((1.0, 1.0))
#     return auc(roc_curve)
#
#
# def roc_through_thresholds(score_label_pairs, threshold_list):
#     scoreidx = 0
#     labelidx = 1
#     look_up = {(True, 1): 'tp', (True, 0): 'fp', (False, 1): 'fn', (False, 0): 'tn'}
#     curve = list()
#     for threshold in threshold_list:
#         counter = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
#         for score_label_pair in score_label_pairs:
#             c = (score_label_pair[scoreidx] >= threshold, score_label_pair[labelidx])
#             counter[look_up[c]] += 1
#         tp = counter['tp']
#         fp = counter['fp']
#         fn = counter['fn']
#         tn = counter['tn']
#         fpr = fp / (fp + tn)
#         tpr = tp / (tp + fn)
#         curve.append((fpr, tpr))
#     return curve
#
#
# def auc(curve, exec_sort=True):
#     if exec_sort:
#         curve = sorted(curve)
#     area = 0.0
#     for idx in range(len(curve) - 1):
#         x1, y1 = curve[idx]
#         x2, y2 = curve[idx + 1]
#         area += (y1 + y2) * (x2 - x1) / 2
#     return area


def roc_auc(lebels, scores):
    return metrics.roc_auc_score(lebels, scores)


def group_array_by_condition(array, item_key):
    dictionary = dict(zip([item_key(item) for item in array], [[] for item in array]))
    for item in array:
        dictionary[item_key(item)].append(item)
    return [dictionary[key] for key in sorted(dictionary.keys())]


def sample_index_by_array_value(array):
    return np.random.choice(a=[i for i in range(len(array))], p=np.array(array) / np.sum(array))


# def softmax(array, factor=1):
#     array = array if factor == 1 else np.array(array) * factor
#     return np.exp(array) / np.sum(np.exp(array), axis=0)

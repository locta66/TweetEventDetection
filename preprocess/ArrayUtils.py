import FileIterator
import numpy as np


def array_partition(array, partition_arr=(1, 1, 1), random=True):
    convert_arr = np.array(array) if not isinstance(array, np.ndarray) else array
    indexes = index_partition(array, partition_arr, random)
    return [convert_arr[indexes[i]] for i in range(len(indexes))]


def arrays_partition(arrays, partition_arr=(1, 1, 1), random=True):
    for i in range(len(arrays) - 1):
        if not len(arrays[i]) == len(arrays[i+1]):
            raise ValueError('Item dimension inconsistent.')
    convert_arr = [np.array(array) if not isinstance(array, np.ndarray) else array for array in arrays]
    print(convert_arr)
    indexes = index_partition(arrays[0], partition_arr, random)
    return [[convert_arr[i][indexes[j]] for i in range(len(arrays))] for j in range(len(indexes))]


def index_partition(array, partition_arr=(1, 1, 1), random=True):
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
    array = array if new_arr else array[:]
    np.random.shuffle(array)
    return array


def roc_auc(score_label_pairs, curve_points=150):
    scoreidx = 0
    score_label_pairs = sorted(score_label_pairs, key=lambda item: item[scoreidx], reverse=True)
    max_threshold = score_label_pairs[0][scoreidx]
    min_threshold = score_label_pairs[-1][scoreidx]
    interval = (max_threshold - min_threshold) / curve_points
    list_of_thresholds = [min_threshold + i * interval for i in range(1, curve_points - 1)]
    
    thresholds_per_process = FileIterator.split_into_multi_format(list_of_thresholds, 15)
    param_list = [(score_label_pairs, process_threshold) for process_threshold in thresholds_per_process]
    roc_curve_arr = FileIterator.multi_process(roc_through_thresholds, param_list)
    roc_curve = FileIterator.merge_list(roc_curve_arr)
    roc_curve.insert(0, (0.0, 0.0))
    roc_curve.append((1.0, 1.0))
    return auc(roc_curve)


def roc_through_thresholds(score_label_pairs, threshold_list):
    scoreidx = 0
    labelidx = 1
    look_up = {(True, 1): 'tp', (True, 0): 'fp', (False, 1): 'fn', (False, 0): 'tn'}
    curve = list()
    for threshold in threshold_list:
        counter = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        for score_label_pair in score_label_pairs:
            c = (score_label_pair[scoreidx] >= threshold, score_label_pair[labelidx])
            counter[look_up[c]] += 1
        tp = counter['tp']
        fp = counter['fp']
        fn = counter['fn']
        tn = counter['tn']
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        curve.append((fpr, tpr))
    return curve


def auc(curve, exec_sort=True, sort_axis=0):
    if exec_sort:
        curve = sorted(curve)
    area = 0.0
    for idx in range(len(curve) - 1):
        x1, y1 = curve[idx]
        x2, y2 = curve[idx + 1]
        area += (y1 + y2) * (x2 - x1) / 2
    return area


def group_array_by_condition(array, item_key):
    dictionary = dict(zip([item_key(item) for item in array], [[] for item in array]))
    for item in array:
        dictionary[item_key(item)].append(item)
    return [dictionary[key] for key in sorted(dictionary.keys())]

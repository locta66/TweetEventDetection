import FunctionUtils as fu

import numpy as np
from sklearn.cluster import dbscan
import Levenshtein

# twarr = fu.load_array('/home/nfs/cdong/tw/testdata/pos_tweets.sum')
# for tw in twarr:
#     tw['temp'] = tw['text'].lower()

# ndim = twarr.__len__()
# mat = np.ones([ndim, ndim])
# for i in range(ndim - 1):
#     for j in range(i + 1, ndim):
#         istr = twarr[i]['temp']
#         jstr = twarr[j]['temp']
#         dist = Levenshtein.distance(istr, jstr) + 1
#         if max(len(istr), len(jstr)) / dist >= 5:
#             mat[i, j] = mat[j, i] = 0
#
# sum(sum(mat))
# label, cluster = dbscan(mat, 0.5, 2, metric='precomputed')
#
# for i in [2, 64, 23]:
#     print(i, end=':')
#     for idx, c in enumerate(cluster):
#         if c == i:
#             print(idx, end=' ')
#     print()
#
# ndim = twarr.__len__()
# mmm = np.ones([ndim, ndim])


def cluster_similar_tweets(twarr):
    if not twarr:
        return twarr
    ndim = len(twarr)
    mat = np.ones([ndim, ndim])
    if len(twarr) <= 128:
        pairs = twarr_dist_pairs(twarr)
    else:
        pairs = twarr_dist_pairs_multi(twarr)
    for p in pairs:
        mat[p[0]][p[1]] = mat[p[1]][p[0]] = p[2]
    label, cluster = dbscan(mat, 0.5, 2, metric='precomputed')
    sort_by_cluster = sorted([(cluster[i], label[i]) for i in range(len(label))])
    return [twarr[sort_by_cluster[i][1]] for i in range(len(sort_by_cluster))]


def twarr_dist_pairs(twarr):
    for tw in twarr:
        tw['nouse'] = tw['text'].lower()
    ndim = len(twarr)
    pairs = list()
    for i in range(ndim - 1):
        for j in range(i + 1, ndim):
            istr = twarr[i]['nouse']
            jstr = twarr[j]['nouse']
            dist = Levenshtein.distance(istr, jstr) + 1
            if max(len(istr), len(jstr)) / dist <= 0.2:
                pairs.append((i, j, 0))
    for tw in twarr:
        del tw['nouse']
    return pairs


def twarr_dist_pairs_multi(twarr):
    for tw in twarr:
        tw['nouse'] = tw['text'].lower()
    total = len(twarr) - 1
    process_num = 16
    point_lists = [[i + 16 * j for j in range(int(total / process_num) + 1)
                    if (i + process_num * j) < total] for i in range(process_num)]
    pairs_blocks = fu.multi_process(dist_pairs, [(twarr, point) for point in point_lists])
    for tw in twarr:
        del tw['nouse']
    return fu.merge_list(pairs_blocks)


def dist_pairs(twarr, points):
    return fu.merge_list([[(i, j, text_dist_less_than(twarr[i]['nouse'], twarr[j]['nouse']))
                           for j in range(i + 1, len(twarr))] for i in points])


def text_dist_less_than(text1, text2, threshold=0.2):
    dist = Levenshtein.distance(text1, text2) + 1
    if dist / max(len(text1), len(text2)) <= threshold:  # text 1 & 2 are similar
        return 0
    else:
        return 1


# ndim = twarr.__len__()
# mmm = np.ones([ndim, ndim])
# pairs = twarr_dist_pairs_multi(twarr)
# for p in pairs:
#     mmm[p[0]][p[1]] = mmm[p[1]][p[0]] = p[2]
#
# sum(sum(mmm))
# label, cluster = dbscan(mmm, 0.5, 2, metric='precomputed')
#
# for i in [2, 64, 23]:
#     print(i, end=':')
#     for idx, c in enumerate(cluster):
#         if c == i:
#             print(idx, end=' ')
#     print()

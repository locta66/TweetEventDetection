import re

from NerServiceProxy import get_ner_service_pool
import FunctionUtils as fu
import DateUtils as du
import TweetKeys as tk

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
    pairs = twarr_dist_pairs(twarr) if len(twarr) <= 128 else twarr_dist_pairs_multi(twarr)
    for p in pairs:
        mat[p[0]][p[1]] = mat[p[1]][p[0]] = p[2]
    label, cluster = dbscan(mat, 0.5, 2, metric='precomputed')
    sort_by_cluster = sorted([(cluster[i], label[i]) for i in range(len(label))])
    return [twarr[sort_by_cluster[i][1]] for i in range(len(sort_by_cluster))]


def twarr_dist_pairs(twarr):
    textarr = [tw['text'].lower() for tw in twarr]
    ndim = len(twarr)
    pairs = list()
    for i in range(ndim - 1):
        for j in range(i + 1, ndim):
            istr, jstr = textarr[i], textarr[j]
            dist = Levenshtein.distance(istr, jstr) + 1
            if max(len(istr), len(jstr)) / dist <= 0.2:
                pairs.append((i, j, 0))
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
    edit_dist = edit_distance(text1, text2) + 1
    return 0 if edit_dist / max(len(text1), len(text2)) <= threshold else 1  # 0 if text 1 & 2 are similar


def edit_distance(text1, text2):
    return Levenshtein.distance(text1, text2)


def twarr_timestamp_array(twarr):
    return [du.get_timestamp_form_created_at(tw['created_at']) for tw in twarr]


def rearrange_idx_by_time(twarr):
    return np.argsort([du.get_timestamp_form_created_at(tw[tk.key_created_at].strip()) for tw in twarr])


def twarr_ner(twarr, using_field=tk.key_text):
    """ Perform NER and POS task upon the twarr, inplace. """
    ner_text_arr = get_ner_service_pool().execute_ner_multiple([tw[using_field] for tw in twarr])
    if not len(ner_text_arr) == len(twarr):
        raise ValueError("Return line number inconsistent; Error occurs during NER")
    for idx, ner_text in enumerate(ner_text_arr):
        wordlabels = parse_ner_text_into_wordlabels(ner_text)
        wordlabels = remove_badword_from_wordlabels(wordlabels)
        twarr[idx][tk.key_wordlabels] = wordlabels
    return twarr


def parse_ner_text_into_wordlabels(ner_text):
    # wordlabels = [('word_0', 'entity extraction 0', 'pos 0'), ('word_1', ...), ...]
    words = re.split('\s+', ner_text)
    wordlabels = list()
    for word in words:
        if word == '':
            continue
        wordlabels.append(parse_ner_word_into_labels(word, slash_num=2))
    return wordlabels


def parse_ner_word_into_labels(ner_word, slash_num):
    """ Split a word into array by '/' searched from the end of the word to its begin.
    :param ner_word: With pos labels.
    :param slash_num: Specifies the number of "/" in the pos word.
    :return: Assume that slash_num=2, "qwe/123"->["qwe","123"], "qwe/123/zxc"->["qwe","123","zxc"],
                              "qwe/123/zxc/456"->["qwe/123","zxc","456"],
    """
    res = list()
    over = False
    for i in range(slash_num):
        idx = ner_word.rfind('/') + 1
        res.insert(0, ner_word[idx:])
        ner_word = ner_word[0:idx - 1]
        if idx == 0:
            over = True
            break
    if not over:
        res.insert(0, ner_word)
    return res


def remove_badword_from_wordlabels(wordlabels):
    for idx, wordlabel in enumerate(wordlabels):
        if re.search('^[^a-zA-Z0-9]+$', wordlabel[0]) is not None:
            del wordlabels[idx]
    return wordlabels

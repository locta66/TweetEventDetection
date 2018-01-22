import re

import utils.function_utils as fu
import utils.date_utils as du
import utils.tweet_keys as tk
import utils.array_utils as au
import utils.spacy_utils as su
import utils.pattern_utils as pu
from utils.ner_service_proxy import get_ner_service_pool

import numpy as np
from sklearn.cluster import dbscan
import Levenshtein


def twarr_similarity(twarr):
    if len(twarr) == 0:
        return None
    if len(twarr) == 1:
        return 1.0
    if tk.key_spacy not in twarr[0]:
        twarr_nlp(twarr)
    """ we have different weight for both word and tweet; for word it is defined by its pos,
        for tweet it is defined by the weight and number of words which have vectors """
    importance_sum = 0
    arr_sum_vec = None
    for tw in twarr:
        tw_ave_vec, vector_num, tw_importance = weighted_doc_ave_vec(tw.get(tk.key_spacy))
        if tw_ave_vec is None or vector_num == 0:
            continue
        tw.setdefault('ave_vec', tw_ave_vec)
        importance_sum += tw_importance
        imp_tw_ave_vec = tw_importance * tw_ave_vec
        arr_sum_vec = (arr_sum_vec + imp_tw_ave_vec) if arr_sum_vec is not None else imp_tw_ave_vec
    
    if arr_sum_vec is not None:
        arr_ave_vec = arr_sum_vec / importance_sum
        return np.mean(au.cosine_similarity(arr_ave_vec.reshape((1, -1)),
                                            [tw.get('ave_vec') for tw in twarr if 'ave_vec' in tw]))
    else:
        return 0


def weighted_doc_ave_vec(doc):
    ent_type_0 = {'NORP', 'FAC', 'GPE', 'LOC', 'ORG', }
    ent_type_1 = {'EVENT', 'PERSON', 'ORDINAL', 'CARDINAL', }
    ent_type_2 = {'DATE', 'TIME', 'PERCENT', }
    ent_weight = dict([(t, 8.0) for t in ent_type_0] + [(t, 2.0) for t in ent_type_1] + [(t, 1.0) for t in ent_type_2])
    # ent_weight = dict([(t, 1.0) for t in ent_type_0] + [(t, 1.0) for t in ent_type_1] + [(t, 1.0) for t in ent_type_2])
    
    tag_type_0 = {'NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBN', 'VBP', 'VBX', }
    tag_type_1 = {'IN', 'JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP', }
    tag_weight = dict([(t, 8.0) for t in tag_type_0] + [(t, 1.0) for t in tag_type_1])
    # tag_weight = dict([(t, 1.0) for t in tag_type_0] + [(t, 1.0) for t in tag_type_1])
    
    doc_vec_num = weight_sum = 0
    doc_sum_vec = None
    for token in doc:
        word, ent_type, tag_type = token.text.lower(), token.ent_type_, token.tag_
        ent_weight_, tag_weight_ = ent_weight.get(ent_type, 1), tag_weight.get(tag_type, 0.1)
        # ent_weight_, tag_weight_ = ent_weight.get(ent_type, 1), tag_weight.get(tag_type, 1.0)
        if not (token.has_vector and pu.is_valid_keyword(word) and
                (ent_type in ent_weight or tag_type in tag_weight)):
            continue
        doc_vec_num += 1
        weight = ent_weight_ * tag_weight_
        weight_sum += weight
        wei_tkn_vec = weight * token.vector
        doc_sum_vec = (doc_sum_vec + wei_tkn_vec) if doc_sum_vec is not None else wei_tkn_vec
    
    doc_ave_vec = (doc_sum_vec / weight_sum) if not doc_vec_num == 0 else None
    return doc_ave_vec, doc_vec_num, (weight_sum / doc_vec_num) if not doc_vec_num == 0 else 0


def twarr_nlp(twarr,
              has_doc=lambda tw: tk.key_spacy in tw,
              get_text=lambda tw: tw.get(tk.key_text),
              nlp_text=lambda text: su.text_nlp(text, su.en_nlp),
              set_doc=lambda tw, doc: tw.setdefault(tk.key_spacy, doc), ):
    for tw in twarr:
        if not has_doc(tw):
            text = get_text(tw)
            doc = nlp_text(text)
            set_doc(tw, doc)
    return twarr


# def twarr_ner(twarr, get_text=lambda tw: tw.get(tk.key_text),
#               set_ner=lambda tw, ner: tw.setdefault(tk.key_ner_pos, ner), nlp=su.en_nlp):
#     for tw in twarr:
#         doc = su.text_nlp(get_text(tw), nlp)
#         ner = [(t.text, t.ent_type_, t.tag_) for t in doc]
#         set_ner(tw, ner)
#     return twarr


def cluster_similar_tweets(twarr):
    if not twarr:
        return twarr
    twarr_length = len(twarr)
    mat = np.ones([twarr_length, twarr_length])
    pairs = twarr_dist_pairs(twarr) if len(twarr) <= 128 else twarr_dist_pairs_multi(twarr)
    for p in pairs:
        mat[p[0]][p[1]] = mat[p[1]][p[0]] = p[2]
    label, cluster = dbscan(mat, 0.5, 2, metric='precomputed')
    sort_by_cluster = sorted([(cluster[i], label[i]) for i in range(len(label))])
    return [twarr[sort_by_cluster[i][1]] for i in range(len(sort_by_cluster))]


def twarr_dist_pairs(twarr):
    textarr = [tw[tk.key_text].lower() for tw in twarr]
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
        tw['nouse'] = tw[tk.key_text].lower()
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
    return [du.get_timestamp_form_created_at(tw[tk.key_created_at]) for tw in twarr]


def rearrange_idx_by_time(twarr):
    return np.argsort([du.get_timestamp_form_created_at(tw[tk.key_created_at].strip()) for tw in twarr])


def start_ner_service(pool_size=8, classify=True, pos=True):
    get_ner_service_pool().start(pool_size, classify, pos)


def end_ner_service():
    get_ner_service_pool().end()


def twarr_ner(twarr, using_field=tk.key_text):
    """ Perform NER and POS task upon the twarr, inplace. """
    ner_text_arr = get_ner_service_pool().execute_nlp_multiple([tw[using_field] for tw in twarr])
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

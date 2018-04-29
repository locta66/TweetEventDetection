from collections import Counter

import utils.function_utils as fu
import utils.timer_utils as tmu
import utils.pattern_utils as pu

import numpy as np


long_word2seg = dict()


def _segment_long_word(word):
    if word not in long_word2seg:
        long_word2seg[word] = pu.segment(word)
    return long_word2seg[word]


def valid_tokens_of_text(text):
    raw_tokens = pu.findall(pu.tokenize_pattern, text)
    seg_tokens = list()
    for token in raw_tokens:
        seg_tokens.extend(_segment_long_word(token)) if len(token) > 16 else seg_tokens.append(token)
    return seg_tokens


def tokens2n_grams(tokens, n):
    if len(tokens) < n:
        return list()
    grams = list()
    if n <= 1:
        for token in tokens:
            if pu.findall(r'[a-zA-Z0-9]+', token) and not pu.is_stop_word(token):
                grams.append(token)
    else:
        for i in range(len(tokens) - n + 1):
            words = ' '.join(tokens[i: i + n])
            grams.append(words)
    return grams


def _get_n2grams(tokens_list, n_range):
    n2grams = dict()
    for n in n_range:
        ngrams = list()
        for tokens in tokens_list:
            ngrams.extend(tokens2n_grams(tokens, n))
        n2grams[n] = Counter(ngrams).most_common()
    return n2grams


def _rescore_grams(n2grams, len_thres, corpus_len):
    # reorder 1,2,3,4 .. grams by their freq/gram, and get top_k
    combine_list = list()
    for n, grams in n2grams.items():
        weight_of_n = np.tanh(corpus_len / 50) * n
        for word, freq in grams:
            if len(word) > len_thres:
                combine_list.append((word, freq * weight_of_n))
    word_score_list = sorted(combine_list, key=lambda item: item[1], reverse=True)
    return word_score_list


def get_quality_keywords(tokens_list, n_range, len_thres):
    n2grams = _get_n2grams(tokens_list, n_range)
    gram_score_list = _rescore_grams(n2grams, len_thres, len(tokens_list))
    keywords = [gram for gram, score in gram_score_list]
    return keywords


def get_quality_n_gram(textarr, n_range, len_thres):
    posttextarr = [text.lower().strip() for text in textarr]
    tokens_list = [valid_tokens_of_text(text) for text in posttextarr]
    keywords = get_quality_keywords(tokens_list, n_range, len_thres)
    return tokens_list, keywords


if __name__ == "__main__":
    # from utils.array_utils import group_textarr_similar_index
    # file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-03-26_suicide-bomb_Lahore.json"
    # twarr = fu.load_array(file)
    # textarr = [tw[tk.key_text] for tw in twarr]
    # tmu.check_time()
    file = "/home/nfs/cdong/tw/src/calling/tmp/0.txt"
    _textarr = fu.read_lines(file)
    _tokens_list = [valid_tokens_of_text(text.lower().strip()) for text in _textarr]
    tmu.check_time()
    _keywords = get_quality_keywords(_tokens_list, n_range=4, len_thres=20, top_k=100)
    tmu.check_time()
    print(_keywords)
    # _ngrams = get_ngrams_from_textarr(_textarr, 4)
    # _reorder_list = reorder_grams(_ngrams, 100)
    # _keywords = [w for w, f in _reorder_list]
    # print(_keywords)
    # idx_g, word_g = group_textarr_similar_index(_keywords, 0.2)
    # for g in word_g:
    #     print(g)

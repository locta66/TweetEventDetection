import numpy as np
from scipy import sparse, io

import utils.tweet_keys as tk
import utils.tweet_utils as tu
import utils.pattern_utils as pu
import utils.function_utils as fu
import utils.spacy_utils as su
from utils.id_freq_dict import IdFreqDict


# 训练推文(不论正负)一次性输入
def freq_count(twarr):
    ifd = IdFreqDict()
    for tw in twarr:
        doc = tw.get(tk.key_spacy)
        for token in doc:
            word = token.text.strip().lower()
            if pu.is_valid_keyword(word):
                ifd.count_word(word)
    ifd.drop_words_by_condition(lambda word, freq: freq*1000< 1000000)
    return ifd, len(twarr)


def feature_matrix_of_twarr(twarr, ifd, doc_num, using_ner=True):
    ifd.reset_id()
    vec_dim = ifd.vocabulary_size()
    idf_lookup = [0] * vec_dim
    for word, freq in ifd.word_freq_enumerate():
        idf_lookup[ifd.word2id(word)] = 10 / np.log((doc_num + 1) / freq)
    matrix = list()
    
    for tw in twarr:
        vector = np.array([0] * vec_dim, dtype=np.float64)
        doc = tw.get(tk.key_spacy)
        added_word = set()
        for token in doc:
            word = token.text.strip().lower()
            if not (pu.is_valid_keyword(word) and ifd.has_word(word)):
                continue
            added_word.add(word)
            wordid = ifd.word2id(word)
            vector[wordid] = idf_lookup[wordid]
        if using_ner:
            vector *= (np.log(len(added_word) + 1) + 1) * (np.log(len(doc.ents) + 1))
        matrix.append(vector)
    
    return np.array(matrix)


def dump_matrix(filename, matrix):
    contract_mtx = sparse.csr_matrix(matrix)
    io.mmwrite(filename, contract_mtx, field='real')


def load_matrix(filename):
    contract_mtx = io.mmread(filename)
    return contract_mtx.todense()


def twarr_nlp(twarr):
    textarr = [tw[tk.key_text] for tw in twarr]
    for idx, doc in enumerate(su.get_nlp().pipe(textarr, n_threads=4)):
        twarr[idx][tk.key_spacy] = doc
    return twarr


if __name__ == '__main__':
    # import time
    # s = time.time()
    # _twarr = fu.load_array('/home/nfs/cdong/tw/testdata/pos_tweets.sum')
    # _textarr = [tw[tk.key_text] for tw in _twarr]
    # for _idx, _doc in enumerate(su.get_nlp().pipe(_textarr, n_threads=4)):
    #     _twarr[_idx][tk.key_spacy] = _doc
    # print('twarr_nlp over, time elapsed {}s'.format(time.time() - s))
    # _ifd, _doc_num = freq_count(_twarr)
    # _mtx = feature_matrix_of_twarr(_twarr, _ifd, _doc_num)
    # dump_matrix('/home/nfs/cdong/tw/testdata/cdong/temp/twarr', _mtx)
    # _mtxx = load_matrix('/home/nfs/cdong/tw/testdata/cdong/temp/twarr')
    #
    
    """"""
    # 假装3年的数据都切分好了
    pos_train_12, neg_train_12 = list(), list()
    pos_test_12, neg_test_12 = list(), list()
    pos_train_16, neg_train_16 = list(), list()
    pos_test_16, neg_test_16 = list(), list()
    pos_train_17, neg_train_17 = list(), list()
    pos_test_17, neg_test_17 = list(), list()
    # 所有推文列表过一遍twarr_nlp，因为是inplace的，可以把列表拼起来放进去提高一点效率 * 12 (train + test)
    twarr_nlp(pos_train_12 + neg_train_12 + ... + neg_test_17)
    # 训练推文(正负)一次性输入，只是用于统计词典 * 6 train
    train = pos_train_12 + pos_train_16 + pos_train_17 + neg_train_12 + neg_train_16 + neg_train_17
    ifd, doc_num = freq_count(train)
    # 构建每个数据集的矩阵，用不用命名实体可以用参数控制，姑且先用着
    pos_train_12_matrix = feature_matrix_of_twarr(pos_train_12, ifd, doc_num, using_ner=True)
    neg_train_12_matrix = feature_matrix_of_twarr(neg_train_12, ifd, doc_num, using_ner=True)
    ...
    neg_test_17_matrix = feature_matrix_of_twarr(neg_test_17, ifd, doc_num, using_ner=True)
    # 保存每个数据集的矩阵
    dump_matrix('文件路径/文件名1', pos_train_12_matrix)
    dump_matrix('文件路径/文件名2', neg_train_12_matrix)
    ...
    dump_matrix('文件路径/文件名n', neg_test_17_matrix)
    
    # 以后读取的时候都用load_matrix来还原矩阵，不再做文本处理
    pos_train_12_matrix = load_matrix('文件路径/文件名1')
    neg_train_12_matrix = load_matrix('文件路径/文件名2')
    ...
    neg_test_17_matrix = load_matrix('文件路径/文件名n')
    # 用矩阵进行训练、测试

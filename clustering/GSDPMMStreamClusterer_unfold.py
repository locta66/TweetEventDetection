from collections import Counter

import numpy as np
import pandas as pd

import ArrayUtils as au
import TweetKeys as tk
import ArkServiceProxy as ark
from IdFreqDict import IdFreqDict
from WordFreqCounter import WordFreqCounter as wfc

np.random.seed(233)

K_POS_TOKEN = tk.key_ark
K_IFD, K_CLUID, K_CLU_COL = 'ifd', 'cluid', 'cluster'
K_ALPHA, K_FREQ_SUM, K_VALID_IFD = 'alpha', 'freqsum', 'valid'
K_FREQ, K_PARAM, K_PARAM0 = 'freq', 'p', 'p0'
LABEL_COLS = {ark.proper_noun_label, ark.common_noun_label, ark.verb_label, ark.hashtag_label}


class GSDPMMStreamClusterer:
    def __init__(self, hold_batch_num=10):
        self.alpha, self.beta = 10, 0.01
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.batch_twnum_list, self.z = list(), list(), list(), list()
        self.max_clu_id = 0
        self.param_alpha = {K_PARAM: 100}
        self.param_table = dict([(label, {K_PARAM: 0.01, K_PARAM0: 0}) for label in LABEL_COLS])
    
    def GSDPMM_twarr(self, old_twarr, old_z, new_twarr, iter_num):
        twarr = old_twarr + new_twarr
        global_label_ifds = dict([(label, IdFreqDict()) for label in LABEL_COLS])
        """pre-process the tweet text"""
        for tw in new_twarr:
            # for every tweet we need to make it clean the first time we meet it
            pos_tokens = tw[K_POS_TOKEN]
            for i in range(len(pos_tokens) - 1, -1, -1):
                word = pos_tokens[i][0] = pos_tokens[i][0].lower().strip()
                if not wfc.is_valid_keyword(word):
                    pos_tokens.pop(i)
        for tw in twarr:
            pos_tokens = tw[K_POS_TOKEN]
            for pos_token in pos_tokens:
                word = pos_token[0]
                real_label = ark.pos_token2label(pos_token)
                if real_label in LABEL_COLS:
                    global_label_ifds[real_label].count_word(word)
        for label in LABEL_COLS:
            global_label_ifds[label].drop_words_by_condition(3)
            param_dict = self.param_table[label]
            param_dict[K_PARAM0] = param_dict[K_PARAM0] * global_label_ifds[label].vocabulary_size()
        for tw in twarr:
            tw.update(dict([(label, IdFreqDict()) for label in LABEL_COLS]))
            pos_tokens = tw[K_POS_TOKEN]
            for pos_token in pos_tokens:
                real_label = ark.pos_token2label(pos_token)
                if real_label in LABEL_COLS:
                    tw[real_label].count(pos_token[0])
        """definitions of parameters"""
        D = len(twarr)
        new_z = [0] * len(new_twarr)
        K = 1 if not old_z else max(old_z) + 1
        cluster_table = [ClusterHolder(i) for i in range(K)]
        """initialize the counting arrays"""
        for old_d in range(len(old_twarr)):
            cluster = old_z[old_d]
            m_z[cluster] += 1
            for word, freq in old_twarr[old_d]['dup'].items():
                n_z[cluster] += freq
                n_zw[cluster][words[word]['id']] += freq
        for new_d in range(len(new_twarr)):
            cluster = int(K * np.random.random())
            new_z[new_d] = cluster
            m_z[cluster] += 1
            for word, freq in new_twarr[new_d]['dup'].items():
                n_z[cluster] += freq
                n_zw[cluster][words[word]['id']] += freq
        """make sampling using current counting information"""
        def sample_cluster(tw, cur_iter=None):
            prob = [0] * K
            tw_freq_dict = tw['dup']
            for k in range(K):
                prob[k] = m_z[k] / (D - 1 + alpha)
                i_ = 0
                for word, freq in tw_freq_dict.items():
                    for j_ in range(freq):
                        prob[k] *= (n_zw[k][words[word]['id']] + beta + j_) / (n_z[k] + beta0 + i_)
                        i_ += 1
            new_cluster_prob = alpha / (D - 1 + alpha)
            i_ = 0
            for word, freq in tw_freq_dict.items():
                for j_ in range(freq):
                    new_cluster_prob *= (beta + j_) / (beta0 + i_)
                    i_ += 1
            if cur_iter is not None and cur_iter > iter_num - 2:
                return np.argmax(prob + [new_cluster_prob])
            else:
                return au.sample_index_by_array_value(np.array(prob + [new_cluster_prob]))
        
        """start iteration"""
        for i in range(iter_num):
            for new_d in range(len(new_twarr)):
                freq_dict = new_twarr[new_d]['dup']
                cluster = new_z[new_d]
                m_z[cluster] -= 1
                for word, freq in freq_dict.items():
                    n_zw[cluster][words[word]['id']] -= freq
                    n_z[cluster] -= freq
                
                min_tw_num_idx = int(np.argmin(m_z))
                if m_z[min_tw_num_idx] == 0:
                    m_z.pop(min_tw_num_idx)
                    n_z.pop(min_tw_num_idx)
                    n_zw.pop(min_tw_num_idx)
                    K -= 1
                    for new_d_ in range(len(new_twarr)):
                        if new_z[new_d_] > min_tw_num_idx:
                            new_z[new_d_] -= 1
                
                cluster = sample_cluster(new_twarr[new_d], i)
                
                if cluster >= K:
                    m_z.append(0)
                    n_z.append(0)
                    n_zw.append([0] * V)
                    K += 1
                
                new_z[new_d] = cluster
                m_z[cluster] += 1
                for word, freq in freq_dict.items():
                    n_z[cluster] += freq
                    n_zw[cluster][words[word]['id']] += freq
        
        return new_z
    
    def input_batch(self, tw_batch, lb_batch=None):
        self.batch_twnum_list.append(len(tw_batch))
        """insufficient tweets"""
        if len(self.batch_twnum_list) < self.hold_batch_num:
            self.twarr += tw_batch
            self.label += lb_batch
            return None, None
        """the first time when len(self.batch_twnum_list) == self.hold_batch_num, may get merged"""
        if not self.init_batch_ready:
            self.init_batch_ready = True
            self.twarr += tw_batch
            self.label += lb_batch
            self.z = self.GSDPMM_twarr(list(), list(), self.twarr, iter_num=30)
            print(' ', dict(Counter(self.z)))
            return list(), self.z[:]
        """normal process of new twarr"""
        old_z = self.z[:]
        new_z = self.GSDPMM_twarr(self.twarr, self.z, tw_batch, iter_num=5)
        
        oldest_twarr_len = self.batch_twnum_list.pop(0)
        popped = list()
        for d in range(oldest_twarr_len):
            popped.append(self.z.pop(0)), self.twarr.pop(0)
        self.twarr += tw_batch
        self.z += new_z
        print(' ', dict(Counter(popped)))
        print(' ', dict(Counter(self.z)))
        return old_z, new_z


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.m = 0
        self.cluster_info_table = pd.DataFrame(columns=LABEL_COLS)
        self.cluster_info_table.loc[K_IFD] = [IdFreqDict() for _ in LABEL_COLS]
        self.cluster_info_table.loc[K_FREQ_SUM] = [0] * len(LABEL_COLS)
    
    def get_ifd_by_label(self, label):
        return self.cluster_info_table.loc[K_IFD, label]
    
    def get_sum_by_label(self, label):
        return self.cluster_info_table.loc[K_FREQ_SUM, label]
    
    def set_sum_increment_by_label(self, label, increment):
        self.cluster_info_table.loc[K_FREQ_SUM, label] += increment

from collections import Counter

import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
from clustering.cluster_service import ClusterService


np.random.seed(233)


class GSDPMMStreamClusterer:
    def __init__(self, hold_batch_num):
        print('using GSDPMMStreamClusterer')
        self.alpha, self.beta = 0.5, 0.01
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.batch_twnum_list, self.z = list(), list(), list(), list()
        self.max_clu_id = 0
    
    def input_batch_with_label(self, tw_batch, lb_batch=None):
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
            self.z = self.GSDPMM_twarr(list(), list(), self.twarr, iter_num=60)
            self.z = [int(item) for item in self.z]
            # print(' ', dict(Counter(self.z)), len(Counter(self.z)))
            return [int(item) for item in self.z], self.label[:]
        """normal process of new twarr"""
        new_z = self.GSDPMM_twarr(self.twarr, self.z, tw_batch, iter_num=4)
        oldest_twarr_len = self.batch_twnum_list.pop(0)
        popped = list()
        for d in range(oldest_twarr_len):
            popped.append(self.z.pop(0)), self.twarr.pop(0), self.label.pop(0)
        self.twarr += tw_batch
        self.label += lb_batch
        self.z += new_z
        # print(' current', dict(Counter(self.z)), len(Counter(self.z)))
        # print(len(self.z))
        return [int(i) for i in self.z], self.label[:]
    
    def GSDPMM_twarr(self, old_twarr, old_z, new_twarr, iter_num):
        pos_token = tk.key_ark
        twarr = old_twarr + new_twarr
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            tokens = tw[pos_token]
            for i in range(len(tokens) - 1, -1, -1):
                tokens[i][0] = tokens[i][0].lower().strip('#').strip()
                if not ClusterService.is_valid_keyword(tokens[i][0]): del tokens[i]
            for wordlabel in tokens:
                word = wordlabel[0]
                if word in words: words[word]['freq'] += 1
                else: words[word] = {'freq': 1, 'id': len(words.keys())}
        min_df = 3
        for w in list(words.keys()):
            if words[w]['freq'] < min_df: del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[pos_token] if wlb[0] in words]))
        """definitions of parameters"""
        D = len(twarr)
        V = len(words.keys())
        alpha, beta = self.alpha, self.beta
        beta0 = V * beta
        new_z = [0] * len(new_twarr)
        K = 1 if not old_z else max(old_z) + 1
        m_z = [0] * K
        n_z = [0] * K
        n_zw = [[0] * V for _ in range(K)]
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
            if cur_iter is not None and cur_iter >= iter_num - 1:
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

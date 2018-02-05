from copy import deepcopy

import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
import utils.ark_service_proxy as ark
from utils.id_freq_dict import IdFreqDict
from clustering.cluster_service import ClusterService


np.random.seed(233)

K_IFD, K_PARAM, K_CLUID, K_CLU_COL = 'ifd', 'param', 'cluid', 'cluster'
K_ALPHA, K_FREQ_SUM, K_VALID_IFD = 'alpha', 'freqsum', 'valid'
K_TW_TABLE = 'table'
LABEL_COLS_LIST = [ark.prop_label, ark.comm_label, ark.verb_label, ark.hstg_label]
LABEL_COLS = set(LABEL_COLS_LIST)


class GSDPMMSemanticStream:
    def __init__(self, hold_batch_num):
        print('using GSDPMMSemanticStreamClusterer')
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.batch_twnum_list, self.z = list(), list(), list(), list()
        self.max_clu_id = 0
        self.global_info_table = dict([(label, IdFreqDict()) for label in LABEL_COLS])
        self.global_valid_table = dict([(label, IdFreqDict()) for label in LABEL_COLS])
        self.cluster_table = {self.max_clu_id: ClusterHolder(self.max_clu_id)}
        self.params_table = dict()
    
    def set_hyperparams(self, alpha, etap, etac, etav, etah):
        self.params_table = dict(zip([K_ALPHA] + LABEL_COLS_LIST,
            [ParamHolder(alpha), ParamHolder(etap), ParamHolder(etac), ParamHolder(etav), ParamHolder(etah)]))
    
    def input_batch_with_label(self, tw_batch, lb_batch):
        self.count_new_batch(tw_batch)
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
            self.z = self.GSDPMM_twarr(list(), self.twarr, iter_num=60)
            # print(' ', dict(Counter(self.z)), len(Counter(self.z)))
            return self.z[:], self.label[:]
        """normal process of new twarr"""
        new_z = self.GSDPMM_twarr(self.twarr, tw_batch, iter_num=8)
        oldest_twarr_len = self.batch_twnum_list.pop(0)
        popped_z = list()
        for d in range(oldest_twarr_len):
            self.update_cluster_with_tw(self.twarr[0][K_CLUID], self.twarr[0], factor=-1)
            popped_z.append(self.z.pop(0)), self.twarr.pop(0), self.label.pop(0)
        self.z += new_z
        self.twarr += tw_batch
        self.label += lb_batch
        # print('popped ', dict(Counter(popped_z)), len(popped_z))
        # print(' current', dict(Counter(self.z)), len(Counter(self.z)))
        return self.z[:], self.label[:]
    
    def GSDPMM_twarr(self, old_twarr, new_twarr, iter_num):
        for i in range(iter_num):
            for idx, tw in enumerate(new_twarr):
                old_cluid = tw[K_CLUID]
                self.update_cluster_sum_valid_with_tw(old_cluid, tw, factor=-1)
                self.update_cluster_with_tw(old_cluid, tw, factor=-1)
                if self.cluster_table[old_cluid].m == 0:
                    self.cluster_table.pop(old_cluid)
                
                new_cluid = self.sample_cluid(tw, len(old_twarr) + len(new_twarr))
                if new_cluid > self.max_clu_id:
                    self.max_clu_id += 1
                    self.cluster_table[self.max_clu_id] = ClusterHolder(self.max_clu_id)
                    self.update_cluster_with_tw(new_cluid, tw, factor=1)
                    self.recount_cluster_sum_valid(new_cluid)
                else:
                    self.update_cluster_sum_valid_with_tw(new_cluid, tw, factor=1)
                    self.update_cluster_with_tw(new_cluid, tw, factor=1)
        
        new_z = [tw[K_CLUID] for tw in new_twarr]
        return new_z
    
    def list_cluid(self): return [idx for idx in self.cluster_table.keys()]
    
    @staticmethod
    def pre_process_twarr(twarr):
        if tk.key_ark not in twarr[0]:
            print('executing pos')
            ark.twarr_ark(twarr)
        for tw in twarr:
            tw.update(dict([(label, IdFreqDict()) for label in LABEL_COLS]))
            pos_tokens = tw[tk.key_ark]
            for pos_token in pos_tokens:
                if not ClusterService.is_valid_keyword(pos_token[0]):
                    continue
                real_label = ark.pos_token2semantic_label(pos_token)
                if real_label in LABEL_COLS:
                    tw[real_label].count_word(pos_token[0].lower().strip())
        return twarr
    
    @staticmethod
    def update_info_table_with_tw(info_table, tw, factor):
        """ Hard update, without considering the problem of validation """
        for label in info_table.keys():
            ifd = info_table[label]
            tw_ifd = tw[label]
            ifd.merge_freq_from(tw_ifd) if factor > 0 else ifd.drop_freq_from(tw_ifd)
    
    def update_cluster_with_tw(self, cluid, tw, factor):
        cluster = self.cluster_table[cluid]
        cluster.m += 1 if factor > 0 else -1
        tw[K_CLUID] = cluid if factor > 0 else -1
        self.update_info_table_with_tw(cluster.cluster_info_table, tw, factor)
    
    def recount_cluster_sum_valid(self, cluid):
        for label in LABEL_COLS:
            cluster = self.cluster_table[cluid]
            g_valid_ifd = self.global_valid_table[label]
            c_ifd = cluster.get_ifd_by_label(label)
            valid_words = set(c_ifd.vocabulary()).intersection(set(g_valid_ifd.vocabulary()))
            valid_sum = sum([c_ifd.freq_of_word(word) for word in valid_words])
            cluster.set_sum_by_label(label, valid_sum)
    
    def update_cluster_sum_valid_with_tw(self, cluid, tw, factor):
        cluster = self.cluster_table[cluid]
        for label in LABEL_COLS:
            g_valid_ifd = self.global_valid_table[label]
            tw_ifd = tw[label]
            for word, freq in tw_ifd.word_freq_enumerate(newest=False):
                if g_valid_ifd.has_word(word):
                    cluster.set_sum_increment_by_label(label, freq if factor > 0 else -freq)
                    # c_ifd = cluster.get_ifd_by_label(label)
                    # print(cluid, sum([c_ifd.freq_of_word(word) for
                    #            word in c_ifd.vocabulary() if g_valid_ifd.has_word(word)]) -
                    #       cluster.get_sum_by_label(label))
                    # if cluster.get_sum_by_label(label) < 0:
                    #     raise ValueError('{} {} {} {} \n {} {} valid {}'.
                    #                      format(word, freq, label, cluster.get_sum_by_label(label),
                    #                             g_valid_ifd.freq_of_word(word), c_ifd.freq_of_word(word),
                    #                             sum([c_ifd.freq_of_word(word) for word in c_ifd.vocabulary()
                    #                                  if g_valid_ifd.has_word(word)])))
    
    def validate_global_ifds(self):
        """ Only alter the global dict while remaining the words in tweets since we make judge afterwards."""
        for label in LABEL_COLS:
            g_valid_ifd = self.global_valid_table[label] = deepcopy(self.global_info_table[label])
            g_valid_ifd.drop_words_by_condition(2)
            self.params_table[label].update_param0(g_valid_ifd.vocabulary_size())
    
    def count_new_batch(self, tw_batch):
        self.pre_process_twarr(tw_batch)
        for tw in tw_batch:
            self.update_info_table_with_tw(self.global_info_table, tw, factor=1)
            self.update_cluster_with_tw(np.random.choice(self.list_cluid()), tw, factor=1)
        self.validate_global_ifds()
        for cluid in self.cluster_table.keys():
            self.recount_cluster_sum_valid(cluid)
    
    def sample_cluid(self, tw, D):
        cluids = self.list_cluid()
        p_alpha = self.params_table[K_ALPHA]
        prob = [0.0] * len(cluids)
        new_clu_prob = p_alpha.param() / (D - 1 + p_alpha.param())
        for label in LABEL_COLS:
            g_valid_ifd = self.global_valid_table[label]
            p_label = self.params_table[label]
            p, p0 = p_label.param(), p_label.param0()
            for k, cluid in enumerate(cluids):
                cluster = self.cluster_table[cluid]
                c_ifd = cluster.get_ifd_by_label(label)
                m_k = cluster.m
                prob[k] = m_k / (D - 1 + p_alpha.param())
                n_z_k = cluster.get_sum_by_label(label)
                i_ = 0
                for word, freq in tw[label].word_freq_enumerate(newest=False):
                    if not g_valid_ifd.has_word(word):
                        continue
                    c_freq_of_word = c_ifd.freq_of_word(word) if c_ifd.has_word(word) else 0
                    for j_ in range(freq):
                        prob[k] *= (c_freq_of_word + p + j_) / (n_z_k + p0 + i_)
                        i_ += 1
            i_ = 0
            for word, freq in tw[label].word_freq_enumerate(newest=False):
                if not g_valid_ifd.has_word(word):
                    continue
                for j_ in range(freq):
                    new_clu_prob *= (p + j_) / (p0 + i_)
                    i_ += 1
        
        cluids.append(self.max_clu_id + 1)
        prob.append(new_clu_prob)
        sample_result = cluids[au.sample_index(np.array(prob))]
        return sample_result


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.m = 0
        self.cluster_info_table = dict([(label, IdFreqDict()) for label in LABEL_COLS])
        self.sum_info_table = dict([(label, 0) for label in LABEL_COLS])
    
    def get_ifd_by_label(self, label): return self.cluster_info_table[label]
    
    def get_sum_by_label(self, label): return self.sum_info_table[label]
    
    def set_sum_by_label(self, label, value): self.sum_info_table[label] = value
    
    def set_sum_increment_by_label(self, label, increment): self.sum_info_table[label] += increment


class ParamHolder:
    def __init__(self, param_value):
        self._param_value = param_value
        self._param0_value = 0
    
    def param(self): return self._param_value
    
    def update_param0(self, factor): self._param0_value = self.param() * factor
    
    def param0(self): return self._param0_value

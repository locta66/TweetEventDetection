import numpy as np

import config.dict_loader as dld
import utils.array_utils as au
import utils.tweet_keys as tk
import utils.ark_service_proxy as ark
from utils.id_freq_dict import IdFreqDict
from clustering.cluster_service import ClusterService


np.random.seed(233)
K_CLUID, K_ALPHA = 'cluid', 'alpha'
DICT_LIST = [dld.prop_dict, dld.comm_dict, dld.verb_dict, dld.hstg_dict]
LABEL_LIST = [ark.prop_label, ark.comm_label, ark.verb_label, ark.hstg_label]
LABEL_SET = set(LABEL_LIST)

global_info_table = dict([(LABEL_LIST[idx], DICT_LIST[idx]) for idx in range(len(DICT_LIST))])


class GSDPMMSemanticStreamStatic:
    def __init__(self, hold_batch_num):
        print('using GSDPMMSemanticStreamStatic')
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.batch_twnum_list, self.z = list(), list(), list(), list()
        self.max_clu_id = 0
        self.cluster_table = {self.max_clu_id: ClusterHolder(self.max_clu_id)}
        self.params_table = dict()
    
    def set_hyperparams(self, alpha, etap, etac, etav, etah):
        self.params_table = dict(zip([K_ALPHA] + LABEL_LIST,
            [ParamHolder(alpha), ParamHolder(etap), ParamHolder(etac), ParamHolder(etav), ParamHolder(etah)]))
        for label in LABEL_LIST:
            self.params_table[label].update_param0(global_info_table[label].vocabulary_size())
    
    def input_batch_with_label(self, tw_batch, lb_batch):
        self.batch_twnum_list.append(len(tw_batch))
        """insufficient tweets"""
        if len(self.batch_twnum_list) < self.hold_batch_num:
            self.count_new_batch(tw_batch, sample=False)
            self.twarr += tw_batch
            self.label += lb_batch
            return None, None
        """the first time when len(self.batch_twnum_list) == self.hold_batch_num, may get merged"""
        if not self.init_batch_ready:
            self.count_new_batch(tw_batch, sample=False)
            self.init_batch_ready = True
            self.twarr += tw_batch
            self.label += lb_batch
            self.z = self.GSDPMM_twarr(list(), self.twarr, iter_num=60)
            # print(' ', dict(Counter(self.z)), len(Counter(self.z)))
            return self.z[:], self.label[:]
        """normal process of new twarr"""
        self.count_new_batch(tw_batch, sample=True)
        new_z = self.GSDPMM_twarr(self.twarr, tw_batch, iter_num=8)
        oldest_len = self.batch_twnum_list.pop(0)
        popped_z = list()
        for d in range(oldest_len):
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
            for tw in new_twarr:
                old_cluid = tw[K_CLUID]
                self.update_cluster_with_tw(old_cluid, tw, factor=-1)
                if self.cluster_table[old_cluid].m == 0:
                    self.cluster_table.pop(old_cluid)
                
                new_cluid = self.sample_cluid(tw, len(old_twarr) + len(new_twarr))
                self.update_cluster_with_tw(new_cluid, tw, factor=1)
        
        new_z = [tw[K_CLUID] for tw in new_twarr]
        return new_z
    
    def count_new_batch(self, tw_batch, sample=False):
        self.pre_process_twarr(tw_batch)
        for tw in tw_batch:
            if sample:
                cluid = self.sample_cluid(tw, len(self.twarr) + len(tw_batch))
            else:
                cluid = np.random.choice(self.list_cluid())
            self.update_cluster_with_tw(cluid, tw, factor=1)
    
    def list_cluid(self):
        return [idx for idx in self.cluster_table.keys()]
    
    @staticmethod
    def pre_process_twarr(twarr):
        if tk.key_ark not in twarr[0]:
            print('executing pos')
            ark.twarr_ark(twarr)
        for tw in twarr:
            for label in LABEL_LIST:
                tw[label] = IdFreqDict()
            pos_tokens = tw[tk.key_ark]
            for pos_token in pos_tokens:
                word = pos_token[0] = pos_token[0].lower().strip()
                if not ClusterService.is_valid_keyword(word):
                    continue
                real_label = ark.pos_token2semantic_label(pos_token)
                if real_label in LABEL_SET and global_info_table[real_label].has_word(word):
                    tw[real_label].count_word(word)
        return twarr
    
    def update_cluster_with_tw(self, cluid, tw, factor):
        """ Takes charge of judging whether to create new cluster given a cluid """
        if cluid > self.max_clu_id:
            if cluid != self.max_clu_id + 1:
                raise ValueError('wrong cluid {} when creating a new cluster'.format(cluid))
            if factor < 0:
                raise ValueError('factor should be non-negative when creating a new cluster')
            cluid = self.max_clu_id = self.max_clu_id + 1
            self.cluster_table[cluid] = ClusterHolder(cluid)
        cluster = self.cluster_table[cluid]
        cluster.m += 1 if factor > 0 else -1
        tw[K_CLUID] = cluid if factor > 0 else -1
        c_info_table = cluster.cluster_info_table
        for label in LABEL_LIST:
            c_ifd = c_info_table[label]
            tw_ifd = tw[label]
            for word, freq in tw_ifd.word_freq_enumerate(newest=False):
                c_ifd.count_word(word, freq) if factor > 0 else c_ifd.uncount_word(word, freq)
                cluster.set_sum_increment_by_label(label, freq if factor > 0 else -freq)
    
    def sample_cluid(self, tw, D):
        cluids = self.list_cluid()
        p_alpha = self.params_table[K_ALPHA]
        prob = [0.0] * len(cluids)
        new_clu_prob = p_alpha.param() / (D - 1 + p_alpha.param())
        for label in LABEL_SET:
            # g_ifd = global_info_table[label]
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
                    c_freq_of_word = c_ifd.freq_of_word(word) if c_ifd.has_word(word) else 0
                    for j_ in range(freq):
                        prob[k] *= (c_freq_of_word + p + j_) / (n_z_k + p0 + i_)
                        i_ += 1
            i_ = 0
            for word, freq in tw[label].word_freq_enumerate(newest=False):
                for j_ in range(freq):
                    new_clu_prob *= (p + j_) / (p0 + i_)
                    i_ += 1
        
        cluids.append(self.max_clu_id + 1)
        prob.append(new_clu_prob)
        sample_result = cluids[au.sample_index_by_array_value(np.array(prob))]
        return sample_result


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.m = 0
        self.cluster_info_table = dict([(label, IdFreqDict()) for label in LABEL_SET])
        self.sum_info_table = dict([(label, 0) for label in LABEL_SET])
    
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

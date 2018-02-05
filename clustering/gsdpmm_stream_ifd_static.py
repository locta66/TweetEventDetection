import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
import utils.tweet_utils as tu
from utils.id_freq_dict import IdFreqDict
from config.dict_loader import token_dict
from clustering.cluster_service import ClusterService


class GSDPMMStreamIFDStatic:
    def __init__(self, hold_batch_num):
        print(self.__class__.__name__)
        self.alpha = self.beta = self.beta0 = None
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.history_len, self.z = list(), list(), list(), list()
        self.max_cluid = 0
        self.cludict = {self.max_cluid: ClusterHolder(self.max_cluid)}
    
    def set_hyperparams(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
        # token_dict().drop_words_by_condition(10)
        # self.beta0 = beta * token_dict().vocabulary_size()
        self.beta0 = beta * 3000
    
    def input_batch_with_label(self, tw_batch, lb_batch=None):
        tw_batch = self.pre_process_twarr(tw_batch)
        self.history_len.append(len(tw_batch))
        if len(self.history_len) < self.hold_batch_num:
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch) if lb_batch is not None else None
            return (None, None) if lb_batch is not None else None
        elif not self.init_batch_ready:
            self.init_batch_ready = True
            self.twarr.extend(tw_batch)
            self.z = self.GSDPMM_twarr(list(), self.twarr, iter_num=30)
            self.label.extend(lb_batch) if lb_batch is not None else None
            return (self.z, self.label[:]) if lb_batch is not None else self.z
        else:
            # since we've made sampling on getting a tweet, not much iteration is needed to make it converge
            new_z = self.GSDPMM_twarr(self.twarr, tw_batch, iter_num=3)
            self.label.extend(lb_batch) if lb_batch is not None else None
            self.twarr.extend(tw_batch)
            self.z.extend(new_z)
            
            oldest_len = self.history_len.pop(0)
            for twh in self.twarr[0: oldest_len]:
                twh.update_cluster(None)
            self.twarr = self.twarr[oldest_len:]
            self.z = self.z[oldest_len:]
            if lb_batch is not None:
                self.label = self.label[oldest_len:]
            return (self.z, self.label[:]) if lb_batch is not None else self.z
    
    @staticmethod
    def pre_process_twarr(twarr):
        twarr = tu.twarr_nlp(twarr)
        return [TweetHolder(tw) for tw in twarr]
    
    def sample(self, twh, D, using_max=False, no_new_clu=False, cluid_range=None):
        alpha = self.alpha
        beta = self.beta
        beta0 = self.beta0
        cludict = self.cludict
        if cluid_range is None:
            cluid_range = cludict.keys()
        cluid_prob = dict()
        for cluid in cluid_range:
            cluster = cludict[cluid]
            clu_tokens = cluster.tokens
            n_zk = clu_tokens.get_freq_sum()
            old_clu_prob = cluster.twnum / (D - 1 + alpha)
            prob_delta = 1.0
            ii = 0
            for word, freq in twh.tokens.word_freq_enumerate(newest=False):
                n_zwk = clu_tokens.freq_of_word(word) if clu_tokens.has_word(word) else 0
                # print('w:{} cid:{} cwf:{} cfs:{}, beta:{} beta0:{}'.format(
                #     word, cluid, clu_word_freq, clu_freq_sum, beta, beta0))
                if freq == 1:
                    prob_delta *= (n_zwk + beta) / (n_zk + beta0 + ii)
                    ii += 1
                else:
                    for jj in range(freq):
                        prob_delta *= (n_zwk + beta + jj) / (n_zk + beta0 + ii)
                        ii += 1
            cluid_prob[cluid] = old_clu_prob * prob_delta
            # print('old:{} init old:{} delta:{}'.format(cluid_prob[cluid], old_clu_prob, prob_delta, ))
        if not no_new_clu:
            ii = 0
            new_clu_prob = alpha / (D - 1 + alpha)
            prob_delta = 1.0
            for word, freq in twh.tokens.word_freq_enumerate(newest=False):
                if freq == 1:
                    prob_delta *= beta / (beta0 + ii)
                    ii += 1
                else:
                    for jj in range(freq):
                        prob_delta *= (beta + jj) / (beta0 + ii)
                        ii += 1
            cluid_prob[self.max_cluid + 1] = new_clu_prob * prob_delta
            # print('new:{} init new:{} delta:{}'.format(cluid_prob[self.max_cluid + 1], new_clu_prob, prob_delta))
            # print('beta:{} beta0:{}'.format(beta, beta0))
            # print()
        
        cluid_arr = list(cluid_prob.keys())
        prob_arr = [cluid_prob[cluid] for cluid in cluid_arr]
        return cluid_arr[np.argmax(prob_arr)] if using_max else cluid_arr[au.sample_index(np.array(prob_arr))]
    
    def resample_sparse_cluster(self, D, is_sparse=lambda twnum: twnum <= 2):
        ii = 0
        for cluid in list(self.cludict.keys()):
            cluster = self.cludict[cluid]
            if is_sparse(cluster.twnum):
                cluid_range = [c for c in self.cludict.keys() if c != cluid]
                for twh in list(cluster.twhdict.values()):
                    # old_cluster = twh.cluster
                    twh.update_cluster(None)
                    new_cluid = self.sample(twh, D, using_max=True, no_new_clu=True, cluid_range=cluid_range)
                    twh.update_cluster(self.cludict[new_cluid])
                    ii += 1
                assert cluster.twnum == 0
                if cluster.twnum == 0:
                    self.cludict.pop(cluster.cluid)
        # print('{} tws resampled'.format(ii))
    
    def GSDPMM_twarr(self, old_twharr, new_twharr, iter_num):
        cludict = self.cludict
        D = len(old_twharr) + len(new_twharr)
        """ allocating cluster for every twh """
        for twh in new_twharr:
            if iter_num <= 20:
                cluid = self.sample(twh, D, using_max=True, no_new_clu=True)
            else:
                cluid = au.choice(list(cludict.keys()))
            twh.update_cluster(cludict[cluid])
        
        """start iteration"""
        for i in range(iter_num):
            # print('iter:{}'.format(i))
            for twh in new_twharr:
                cluster = twh.cluster
                twh.update_cluster(None)
                
                if cluster.twnum == 0:
                    cludict.pop(cluster.cluid)
                
                cluid = self.sample(twh, D, using_max=False)
                if cluid > self.max_cluid:
                    self.max_cluid += 1
                    cludict[cluid] = ClusterHolder(self.max_cluid)
                    cluid = self.max_cluid
                
                twh.update_cluster(cludict[cluid])
        
        # for cluid in list(self.cludict.keys()):
        #     print('clean')
        #     if self.cludict[cluid].twnum == 0:
        #         self.cludict.pop(cluid)
        # print('total tw in sets {}'.format(sum([len(retwset.twhdict) for retwset in self.retwsetdict.values()])))
        # print('total tw by clusters:{}'.format(sum([cluster.twnum for cluster in self.cludict.values()])))
        # print('cluster number:{}'.format(len(self.cludict.keys())))
        # print('clu twnum distrb:{}'.format([cluster.twnum for cluster in self.cludict.values()]))
        # print(sorted(self.cludict.keys()))
        # # print(self.cludict[0].tokens.word_freq_enumerate())
        
        self.resample_sparse_cluster(D)
        
        for cluid in list(self.cludict.keys()):
            if self.cludict[cluid].twnum == 0:
                self.cludict.pop(cluid)
        
        return [int(twh.get_cluid()) for twh in new_twharr]
    
    def get_hyperparams_info(self):
        return 'GSDPMMStreamIFDStatic, alpha={:<5}, beta={:<5}'.format(self.alpha, self.beta)
    
    def clusters_similarity(self):
        return ClusterService.cluster_inner_similarity(self.twarr, self.z)


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.twhdict = dict()
        self.tokens = IdFreqDict()
        self.twnum = 0
    
    def update_by_twh(self, twh, factor):
        twh_tokens = twh.tokens
        twh_id = twh.id
        if factor > 0:
            self.tokens.merge_freq_from(twh_tokens)
            self.twhdict[twh_id] = twh
            self.twnum += 1
        else:
            self.tokens.drop_freq_from(twh_tokens)
            self.twhdict.pop(twh_id)
            self.twnum -= 1


class TweetHolder:
    def __init__(self, tw):
        self.tw = tw
        self.id = tw.get(tk.key_id)
        self.tokens = None
        self.cluster = None
        self.tokenize()
    
    def __contains__(self, key): return key in self.tw
    
    def __getitem__(self, key): return self.get(key)
    
    def __setitem__(self, key, value): self.setdefault(key, value)
    
    def get(self, key): return self.tw.get(key, None)
    
    def setdefault(self, key, value): self.tw.setdefault(key, value)
    
    def get_cluid(self):
        if self.cluster is None:
            raise ValueError('cluster should not be None')
        return self.cluster.cluid
    
    def tokenize(self):
        self.tokens = IdFreqDict()
        for token in self.tw[tk.key_spacy]:
            word = token.text.lower().strip('#').strip()
            if ClusterService.is_valid_keyword(word) and token_dict().has_word(word):
                self.tokens.count_word(word)
    
    def update_cluster(self, cluster):
        if self.cluster is not None:
            self.cluster.update_by_twh(self, factor=-1)
        self.cluster = cluster
        if self.cluster is not None:
            self.cluster.update_by_twh(self, factor=1)

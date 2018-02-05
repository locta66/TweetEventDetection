import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
import utils.tweet_utils as tu
from utils.id_freq_dict import IdFreqDict
from config.dict_loader import token_dict
from clustering.cluster_service import ClusterService


class GSDPMMStreamIFDDynamic:
    def __init__(self, hold_batch_num):
        print(self.__class__.__name__)
        self.alpha = self.beta = self.beta0 = None
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.history_len, self.z = list(), list(), list(), list()
        self.max_cluid = 0
        self.cludict = {self.max_cluid: ClusterHolder(self.max_cluid)}
        self.valid_dict = IdFreqDict()
    
    def set_hyperparams(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
    
    def input_batch_with_label(self, tw_batch, lb_batch):
        self.history_len.append(len(tw_batch))
        tw_batch = self.pre_process_twarr(tw_batch, lb_batch)
        if len(self.history_len) < self.hold_batch_num:
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch)
            return None, None
        elif not self.init_batch_ready:
            self.init_batch_ready = True
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch)
            self.GSDPMM_twarr(list(), self.twarr, iter_num=30)
            return self.get_z(), self.get_label()
        else:
            self.GSDPMM_twarr(self.twarr, tw_batch, iter_num=3)
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch)
            
            oldest_len = self.history_len.pop(0)
            for twh in self.twarr[0: oldest_len]:
                twh.update_cluster(None)
            self.twarr = self.twarr[oldest_len:]
            return self.get_z, self.get_label()
    
    def get_z(self):
        z = list()
        for twh in self.twarr:
            if twh.cluster is None or twh.get_cluid() not in self.cludict:
                raise ValueError('wrong cluid for twh')
            z.append(int(twh.get_cluid()))
        return z
    
    def get_label(self):
        return [int(twh.label) for twh in self.twarr]
    
    @staticmethod
    def pre_process_twarr(twarr, lbarr):
        # every tw should only get processed once
        twarr = tu.twarr_nlp(twarr)
        twharr = list()
        for idx in range(len(twarr)):
            tw, lb = twarr[idx], lbarr[idx]
            twh = TweetHolder(tw)
            twh.label = lb
            twharr.append(twh)
        return twharr
    
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
        sampled_idx = np.argmax(prob_arr) if using_max else au.sample_index(np.array(prob_arr))
        return cluid_arr[sampled_idx]
    
    def GSDPMM_twarr(self, old_twharr, new_twharr, iter_num):
        cludict = self.cludict
        if len(old_twharr) > 0:
            for cluster in self.cludict.values():
                cluster.clear()
        D = len(old_twharr) + len(new_twharr)
        """ redo the dictionary """
        valid_dict = self.valid_dict
        valid_dict.clear()
        for twh in old_twharr + new_twharr:
            valid_dict.merge_freq_from(twh.tokens)
        valid_dict.drop_words_by_condition(3)
        """ reallocate """
        for old_twh in old_twharr:
            old_twh.validate(valid_dict)
            old_cluid = old_twh.get_cluid()
            if old_cluid not in cludict:
                raise ValueError('cluid {} should be in cludict'.format(old_cluid))
            old_twh.cluster = None
            old_twh.update_cluster(cludict[old_cluid])
        for new_twh in new_twharr:
            new_twh.validate(valid_dict)
            if len(old_twharr) > 0:
                new_cluid = self.sample(new_twh, D, using_max=True, no_new_clu=True)
            else:
                new_cluid = self.max_cluid
            new_twh.update_cluster(cludict[new_cluid])
        """ params """
        self.beta0 = self.beta * valid_dict.vocabulary_size()
        # print(self.beta0, valid_dict.vocabulary_size())
        """ start iteration """
        for i in range(iter_num):
            # print('iter:{}'.format(i))
            for twh in new_twharr:
                cluster = twh.cluster
                twh.update_cluster(None)
                
                if cluster.twnum == 0:
                    assert len(cluster.twhdict) == 0
                    cludict.pop(cluster.cluid)
                
                cluid = self.sample(twh, D, using_max=(i == iter_num-1))
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
        
        for cluid in list(self.cludict.keys()):
            if self.cludict[cluid].twnum == 0:
                assert len(self.cludict[cluid].twhdict) == 0
                self.cludict.pop(cluid)
        
        for cluid, cluster in self.cludict.items():
            if cluster.twnum == 0 or len(cluster.twhdict) == 0:
                raise ValueError('empty cluster')
        
        # return [int(twh.get_cluid()) for twh in new_twharr]
    
    def get_hyperparams_info(self):
        return 'GSDPMMStreamIFDDynamic, alpha={:<5}, beta={:<5}'.format(self.alpha, self.beta)
    
    def clusters_similarity(self):
        return ClusterService.cluster_inner_similarity(self.twarr, self.z)


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.twhdict = dict()
        self.tokens = IdFreqDict()
        self.entifd = IdFreqDict()
        self.twnum = 0
    
    def clear(self):
        self.twhdict.clear()
        self.tokens.clear()
        self.entifd.clear()
        self.twnum = 0
    
    def extract_ents(self):
        self.entifd.clear()
        for twh in self.twhdict.values():
            doc = twh.get(tk.key_spacy)
            for ent in doc.ents:
                ent_text = ent.text.strip().lower()
                self.entifd.count_word(ent_text)
    
    def update_by_twh(self, twh, factor):
        twh_tokens = twh.valid_tokens
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
        self.label = None
        self.tokens = IdFreqDict()
        self.valid_tokens = IdFreqDict()
        self.cluster = None
        self.tokenize(token_dict())
    
    def __contains__(self, key): return key in self.tw
    
    def __getitem__(self, key): return self.get(key)
    
    def __setitem__(self, key, value): self.setdefault(key, value)
    
    def get(self, key): return self.tw.get(key, None)
    
    def setdefault(self, key, value): self.tw.setdefault(key, value)
    
    def get_cluid(self): return self.cluster.cluid
    
    def tokenize(self, using_ifd):
        self.tokens = IdFreqDict()
        for token in self.tw[tk.key_spacy]:
            # word = token.text.lower().strip('#').strip()
            word = token.text.lower().strip()
            if ClusterService.is_valid_keyword(word) and using_ifd.has_word(word):
                self.tokens.count_word(word)
    
    def validate(self, using_ifd):
        self.valid_tokens.clear()
        for word, freq in self.tokens.word_freq_enumerate(newest=False):
            if using_ifd.has_word(word):
                self.valid_tokens.count_word(word, freq)
    
    def update_cluster(self, cluster):
        if self.cluster is not None:
            self.cluster.update_by_twh(self, factor=-1)
        self.cluster = cluster
        if cluster is not None:
            cluster.update_by_twh(self, factor=1)

import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
import utils.tweet_utils as tu
from utils.id_freq_dict import IdFreqDict
from config.dict_loader import token_dict
from clustering.cluster_service import ClusterService


np.random.seed(233333)


class GSDPMMStreamRetweet:
    def __init__(self, hold_batch_num):
        print('GSDPMMStreamRetweetClusterer')
        self.alpha = self.beta = self.beta0 = None
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.history_len, self.z = list(), list(), list(), list()
        self.max_cluid = 0
        self.cludict = {self.max_cluid: ClusterHolder(self.max_cluid)}
        self.retwsetdict = dict()
    
    def set_hyperparams(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
        token_dict().drop_words_by_condition(10)
        # self.beta0 = beta * token_dict().vocabulary_size()
        self.beta0 = beta * 10000
    
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
            self.z = self.GSDPMM_twarr(list(), self.twarr, iter_num=40)
            self.label.extend(lb_batch) if lb_batch is not None else None
            return (self.z, self.label[:]) if lb_batch is not None else self.z
        else:
            new_z = self.GSDPMM_twarr(self.twarr, tw_batch, iter_num=5)
            self.label.extend(lb_batch) if lb_batch is not None else None
            self.twarr.extend(tw_batch)
            self.z.extend(new_z)
            
            oldest_len = self.history_len.pop(0)
            for twh in self.twarr[0: oldest_len]:
                twh.abandon()
            self.twarr = self.twarr[oldest_len:]
            self.z = self.z[oldest_len:]
            if lb_batch is not None:
                self.label = self.label[oldest_len:]
            return (self.z, self.label[:]) if lb_batch is not None else self.z
    
    @staticmethod
    def pre_process_twarr(twarr):
        twarr = tu.twarr_nlp(twarr)
        return [TweetHolder(tw) for tw in twarr]
    
    def sample(self, retwset, D, using_max=False, no_new_clu=False):
        assert len(retwset.get_twharr()) > 0
        prob_twset = []
        alpha, beta, beta0 = self.alpha, self.beta, self.beta0
        for twh in retwset.get_twharr():
            cluid_prob = dict()
            for cluid, cluster in self.cludict.items():
                m_zk, clu_tokens = cluster.twnum, cluster.tokens
                clu_freq_sum = clu_tokens.get_freq_sum()
                old_clu_prob = m_zk / (D - 1 + alpha)
                prob_delta = 1.0
                ii = 0
                for word, freq in twh.tokens.word_freq_enumerate(newest=False):
                    clu_word_freq = clu_tokens.freq_of_word(word) if clu_tokens.has_word(word) else 0
                    # print('w:{} cid:{} cwf:{} cfs:{}, beta:{} beta0:{}'.format(
                    #     word, cluid, clu_word_freq, clu_freq_sum, beta, beta0))
                    for jj in range(freq):
                        prob_delta *= (clu_word_freq + beta + jj) / (clu_freq_sum + beta0 + ii)
                        ii += 1
                cluid_prob[cluid] = old_clu_prob * prob_delta
                # print('old:{} init old:{} delta:{}'.format(cluid_prob[cluid], old_clu_prob, prob_delta))
            ii = 0
            new_clu_prob = alpha / (D - 1 + alpha)
            prob_delta = 1.0
            for word, freq in twh.tokens.word_freq_enumerate(newest=False):
                for jj in range(freq):
                    prob_delta *= (beta + jj) / (beta0 + ii)
                    ii += 1
            cluid_prob[self.max_cluid + 1] = new_clu_prob * prob_delta
            # print('new:{} init new:{} delta:{}'.format(cluid_prob[self.max_cluid + 1], new_clu_prob, prob_delta))
            # print()
            if no_new_clu:
                cluid_prob.pop(self.max_cluid + 1)
            
            cluid_arr = sorted(cluid_prob.keys())
            prob_arr = [cluid_prob[__cluid] for __cluid in cluid_arr]
            # print(prob_arr)
            pred_cluid = cluid_arr[np.argmax(prob_arr)] if using_max else cluid_arr[au.sample_index(np.array(prob_arr))]
            prob_twset.append(pred_cluid)
        
        if len(retwset.get_twharr()) > 1:
            final_sample = int(au.choice(prob_twset))
        else:
            final_sample = prob_twset[0]
        return final_sample
    
    def GSDPMM_twarr(self, old_twharr, new_twharr, iter_num):
        D = len(old_twharr) + len(new_twharr)
        # print('D', D)
        for twh in new_twharr:
            twid, retwid = twh.id, twh.retwid
            if retwid is not None and retwid not in self.retwsetdict:  # in reply and the target is absent
                retwset = self.retwsetdict[retwid] = RetweetSet(twh, is_master=False)
            elif retwid is not None and retwid in self.retwsetdict:  # in reply and the target is present
                retwset = self.retwsetdict[retwid]
            elif twid is not None and twid not in self.retwsetdict:  # no reply, create a new set
                retwset = self.retwsetdict[twid] = RetweetSet(twh, is_master=True)
            elif twid is not None and twid in self.retwsetdict:  # no reply, and the set has been created
                retwset = self.retwsetdict[twid]
            else:
                raise ValueError('tweet information needed')
            twh.into_retwset(retwset)
        
        noclu_retwset = dict([(setid, self.retwsetdict[setid]) for setid in self.retwsetdict.keys()
                              if self.retwsetdict[setid].cluster is None])
        for setid, retwset in noclu_retwset.items():
            cluid = self.sample(retwset, D, using_max=True, no_new_clu=True) \
                if iter_num <= 20 else au.choice(list(self.cludict.keys()))
            retwset.set_cluster(self.cludict[cluid])
        """start iteration"""
        for i in range(iter_num):
            for setid, retwset in noclu_retwset.items():
                assert retwset.cluster is not None
                old_cluid = retwset.get_cluid()
                old_cluster = self.cludict[old_cluid]
                retwset.set_cluster(None)
                
                if old_cluster.twnum == 0:
                    for twh in old_twharr + new_twharr:
                        if twh.retwset.cluster is not None:
                            assert twh.get_cluid() != old_cluster
                    self.cludict.pop(old_cluid)
                
                cluid = self.sample(retwset, D, using_max=(i >= iter_num - 1))
                if cluid > self.max_cluid:
                    cluster = ClusterHolder(cluid)
                    self.cludict[cluid] = cluster
                    self.max_cluid += 1
                else:
                    cluster = self.cludict[cluid]
                
                retwset.set_cluster(cluster)
        
        for cluid in list(self.cludict.keys()):
            if self.cludict[cluid].twnum == 0:
                self.cludict.pop(cluid)
        
        # print('total tw in sets {}'.format(sum([len(retwset.twhdict) for retwset in self.retwsetdict.values()])))
        # print('total tw by clusters:{}'.format(sum([cluster.twnum for cluster in self.cludict.values()])))
        # print('cluster number:{}'.format(len(self.cludict.keys())))
        # print('clu twnum distrb:{}'.format([cluster.twnum for cluster in self.cludict.values()]))
        # print(sorted(self.cludict.keys()))
        # # print(self.cludict[0].tokens.word_freq_enumerate())
        
        # for setid, retwset in self.retwsetdict.items():
        #     if len(retwset.get_twharr()) >= 2:
        #         print(retwset.get_twharr())
        # for idx, twh in enumerate(new_twharr):
        #     if twh.retwset is None:
        #         raise ValueError('is None')
        
        return [int(twh.get_cluid()) for twh in new_twharr]
    
    def get_hyperparams_info(self): return 'GSDPMM,stream, alpha={:<5}, beta={:<5}'.format(self.alpha, self.beta)
    
    def clusters_similarity(self): return ClusterService.cluster_inner_similarity(self.twarr, self.z)


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.retwsetdict = dict()
        self.tokens = IdFreqDict()
        self.twnum = 0
    
    def update_by_retwset(self, retwset, factor):
        set_mstid = retwset.master_twhid
        if factor > 0:
            # if retw_mstid in self._retwsetdict:
            #     raise ValueError('cannot move in retwset since retwid {} is in cluster'.format(retw_mstid))
            self.retwsetdict[set_mstid] = retwset
            for twh in retwset.get_twharr():
                self.update_by_twh(twh, 1)
        else:
            # if retw_mstid not in self._retwsetdict:
            #     raise ValueError('cannot move out retwset since retwid {} not in cluster'.format(retw_mstid))
            for twh in retwset.get_twharr():
                self.update_by_twh(twh, -1)
            self.retwsetdict.pop(set_mstid)
    
    def update_by_twh(self, twh, factor):
        twh_tokens = twh.tokens
        if factor > 0:
            self.tokens.merge_freq_from(twh_tokens)
            self.twnum += 1
        else:
            self.tokens.drop_freq_from(twh_tokens)
            self.twnum -= 1


class RetweetSet:
    def __init__(self, twh, is_master):
        self.twhdict = dict()
        self.cluster = None
        twid, retwid = twh.id, twh.retwid
        self.master_twhid = (twid if is_master else retwid)
        self.twhdict.setdefault(twid, twh)
    
    def get_twharr(self): return self.twhdict.values()
    
    def get_cluid(self): return self.cluster.cluid
    
    def set_cluster(self, cluster):
        if self.cluster is not None:
            self.cluster.update_by_retwset(self, -1)
        self.cluster = cluster
        if self.cluster is not None:
            self.cluster.update_by_retwset(self, 1)
    
    # def can_join_twh(self, twh):
    #     return twh.get_retwid() is not None and \
    #            (twh.get_retwid() == self.master_twhid or twh.get_id() == self.master_twhid)
    
    def move_twh_into_cluster(self, twh):
        # if self.can_join_twh(twh):
        #     if factor < 0:
        #         assert len(self.twhdict) == 1
        twid = twh.id
        self.twhdict[twid] = twh
        if self.cluster is not None:
            self.cluster.update_by_twh(twh, factor=1)
    
    def remove_twh_from_cluster(self, twh):
        if self.cluster is not None:
            self.cluster.update_by_twh(twh, factor=-1)


class TweetHolder:
    def __init__(self, tw):
        self.tw = tw
        self.id, self.retwid = tw.get(tk.key_id), tu.in_reply_to(tw)
        self.tokens = None
        self.retwset = None
        self.tokenize()
    
    def __getitem__(self, key): return self.get(key)
    
    def __setitem__(self, key, value): self.setdefault(key, value)
    
    def get(self, key): return self.tw.get(key, None)
    
    def setdefault(self, key, value): self.tw.setdefault(key, value)
    
    # def get_id(self): return self.id
    #
    # def get_retwid(self): return self.retwid
    
    def get_cluid(self):
        if self.retwset is None:
            raise ValueError('_retwset in twh should not be None when getting cluid')
        return self.retwset.get_cluid()
    
    def tokenize(self):
        self.tokens = IdFreqDict()
        for token in self.tw[tk.key_spacy]:
            word = token.text.lower().strip('#').strip()
            if ClusterService.is_valid_keyword(word) and token_dict().has_word(word):
                self.tokens.count_word(word)
    
    def into_retwset(self, retwset):
        # if retwset is not None and retwset.can_join_twh(self):
        # if self._retwset is not None:
        #     self._retwset.update_by_twh(self, factor=-1)
        self.retwset = retwset
        self.retwset.move_twh_into_cluster(self)
    
    def abandon(self):
        self.retwset.remove_twh_from_cluster(self)

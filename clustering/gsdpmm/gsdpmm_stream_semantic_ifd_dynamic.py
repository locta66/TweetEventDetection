import numpy as np

import utils.pattern_utils as pu
import utils.spacy_utils as su
import utils.array_utils as au
import utils.tweet_keys as tk
import utils.tweet_utils as tu
from utils.id_freq_dict import IdFreqDict
import clustering.cluster_service as cs


class GSDPMMStreamSemanticIFDDynamic:
    """ Dynamic cluster number, stream, dynamic dictionary, semantic, using ifd """
    
    def __init__(self, hold_batch_num):
        print(self.__class__.__name__)
        self.alpha = None
        self.p_dict = dict([(k_type, None) for k_type in TokenSet.KEY_LIST])
        self.p0_dict = dict([(k_type, None) for k_type in TokenSet.KEY_LIST])
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.history_len, self.z = list(), list(), list(), list()
        self.max_cluid = 0
        self.cludict = {self.max_cluid: ClusterHolder(self.max_cluid)}
        self.valid_corpus_token_set = TokenSet()
    
    def set_hyperparams(self, alpha, etap, etac, etav, etah):
        self.alpha = alpha
        self.p_dict[su.pos_prop] = etap
        self.p_dict[su.pos_comm] = etac
        self.p_dict[su.pos_verb] = etav
        self.p_dict[su.pos_hstg] = etah
    
    def input_batch_with_label(self, tw_batch, lb_batch):
        tw_batch = self.pre_process_twarr(tw_batch, lb_batch)
        self.history_len.append(len(tw_batch))
        if len(self.history_len) < self.hold_batch_num:
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch)
            return None, None
        elif not self.init_batch_ready:
            self.init_batch_ready = True
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch)
            self.z = self.GSDPMM_twarr(list(), self.twarr, iter_num=30)
            # self.clusters_semantic_similarity('sim_info.txt')
            return self.get_z(), self.get_label()
        else:
            new_z = self.GSDPMM_twarr(self.twarr, tw_batch, iter_num=3)
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch)
            self.z.extend(new_z)
            # self.clusters_semantic_similarity('sim_info.txt')
            oldest_len = self.history_len.pop(0)
            for twh in self.twarr[:oldest_len]:
                twh.update_cluster(None)
            self.check_and_drop()
            self.twarr = self.twarr[oldest_len:]
            # self.z = self.z[oldest_len:]
            # self.label = self.label[oldest_len:]
            return self.get_z(), self.get_label()
    
    def get_z(self):
        z = list()
        for twh in self.twarr:
            if twh.cluster is None or twh.get_cluidarr() not in self.cludict:
                raise ValueError('wrong cluid for twh')
            z.append(int(twh.get_cluidarr()))
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
    
    @staticmethod
    def old_prob_delta(twh_ifd, clu_ifd, p, p0):
        prob_delta = 1.0
        if p0 == 0:
            p0 = 10000 * p
        ii = 0
        n_zk = clu_ifd.get_freq_sum()
        for word, freq in twh_ifd.word_freq_enumerate(newest=False):
            n_zwk = clu_ifd.freq_of_word(word) if clu_ifd.has_word(word) else 0
            if freq == 1:
                prob_delta *= (n_zwk + p) / (n_zk + p0 + ii)
                ii += 1
            else:
                for jj in range(freq):
                    prob_delta *= (n_zwk + p + jj) / (n_zk + p0 + ii)
                    ii += 1
        return prob_delta
    
    @staticmethod
    def new_prob_delta(twh_ifd, p, p0):
        prob_delta = 1.0
        ii = 0
        for word, freq in twh_ifd.word_freq_enumerate(newest=False):
            if freq == 1:
                prob_delta *= p / (p0 + ii)
                ii += 1
            else:
                for jj in range(freq):
                    prob_delta *= (p + jj) / (p0 + ii)
                    ii += 1
        return prob_delta
    
    def check_and_drop(self):
        cludict = self.cludict
        for cluid in list(cludict.keys()):
            if cludict[cluid].twnum == 0 or len(cludict[cluid].twhdict) == 0:
                cludict.pop(cluid)
    
    def sample(self, twh, D, using_max=False, no_new_clu=False, cluid_range=None):
        alpha = self.alpha
        p_dict = self.p_dict
        p0_dict = self.p0_dict
        cludict = self.cludict
        tw_valid_token_set = twh.valid_token_set
        etap, etac, etav, etah = [p_dict.get(k_type) for k_type in TokenSet.KEY_LIST]
        etap0, etac0, etav0, etah0 = [p0_dict.get(k_type) for k_type in TokenSet.KEY_LIST]
        twhp, twhc, twhv, twhh = [tw_valid_token_set.get(k_type) for k_type in TokenSet.KEY_LIST]
        
        if cluid_range is None:
            cluid_range = cludict.keys()
        
        cluid_prob = dict()
        old_prob_delta = self.old_prob_delta
        new_prob_delta = self.new_prob_delta
        
        for cluid in cluid_range:
            cluster = cludict[cluid]
            clu_token_set = cluster.token_set
            old_clu_prob = cluster.twnum / (D - 1 + alpha)
            prob_delta = 1.0
            prob_delta *= old_prob_delta(twhp, clu_token_set.get(su.pos_prop), etap, etap0)
            prob_delta *= old_prob_delta(twhc, clu_token_set.get(su.pos_comm), etac, etac0)
            prob_delta *= old_prob_delta(twhv, clu_token_set.get(su.pos_verb), etav, etav0)
            prob_delta *= old_prob_delta(twhh, clu_token_set.get(su.pos_hstg), etah, etah0)
            cluid_prob[cluid] = old_clu_prob * prob_delta
        
        if not no_new_clu:
            new_clu_prob = alpha / (D - 1 + alpha)
            prob_delta = 1.0
            prob_delta *= new_prob_delta(twhp, etap, etap0)
            prob_delta *= new_prob_delta(twhc, etac, etac0)
            prob_delta *= new_prob_delta(twhv, etav, etav0)
            prob_delta *= new_prob_delta(twhh, etah, etah0)
            cluid_prob[self.max_cluid + 1] = new_clu_prob * prob_delta
            # print('new:{} init new:{} delta:{}'.format(cluid_prob[self.max_cluid + 1], new_clu_prob, prob_delta))
            # print()
        
        cluid_arr = list(cluid_prob.keys())
        prob_arr = [cluid_prob[cluid] for cluid in cluid_arr]
        sampled_idx = np.argmax(prob_arr) if using_max else au.sample_index(np.array(prob_arr))
        return cluid_arr[sampled_idx]
    
    def GSDPMM_twarr(self, old_twharr, new_twharr, iter_num):
        cludict = self.cludict
        valid_corpus_token_set = self.valid_corpus_token_set
        if len(old_twharr) > 0:
            for cluster in self.cludict.values():
                cluster.clear()
        D = len(old_twharr) + len(new_twharr)
        """ redo the dictionary and reallocate """
        valid_corpus_token_set.clear()
        for twh in old_twharr + new_twharr:
            valid_corpus_token_set.merge_freq_from(twh.orgn_token_set)
        valid_corpus_token_set.drop_words_by_condition()
        for old_twh in old_twharr:
            old_twh.validate_tokenize(valid_corpus_token_set)
            old_cluid = old_twh.get_cluidarr()
            assert old_cluid in cludict
            old_twh.cluster = None
            old_twh.update_cluster(cludict[old_cluid])
        for new_twh in new_twharr:
            new_twh.validate_tokenize(valid_corpus_token_set)
            if len(old_twharr) > 0:
                new_cluid = self.sample(new_twh, D, using_max=True, no_new_clu=True)
            else:
                new_cluid = self.max_cluid
            new_twh.update_cluster(cludict[new_cluid])
        """ params update """
        valid_ifd_dict = valid_corpus_token_set.type_ifd_dict
        for k_type in TokenSet.KEY_LIST:
            self.p0_dict[k_type] = self.p_dict[k_type] * valid_ifd_dict[k_type].vocabulary_size()
        """ start iteration """
        for i in range(iter_num):
            # print('iter:{}'.format(i))
            for twh in new_twharr:
                cluster = twh.cluster
                twh.update_cluster(None)
                
                if cluster.twnum == 0:
                    assert len(cluster.twhdict) == 0
                    cludict.pop(cluster.cluid)
                
                cluid = self.sample(twh, D, using_max=(i == iter_num - 1))
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
        return [int(twh.get_cluidarr()) for twh in new_twharr]
    
    def get_hyperparams_info(self):
        return 'GSDPMMStreamSemanticDynamic, alpha={:<5}, param dict={}'.format(self.alpha, self.p_dict)
    
    def clusters_tfidf_similarity(self, file_name):
        import pandas as pd
        import utils.function_utils as fu
        # print('clusters_tfidf_similarity')
        # tmu.check_time()
        # """ construct tf vector for every cluster """
        # cluid_arr = sorted(self.cludict.keys())
        # valid_corpus_token_set = self.valid_corpus_token_set
        # cluid2vec = dict([(cluid, None) for cluid in cluid_arr])
        # for cluid, cluster in self.cludict.items():
        #     if cluster.twnum == 0:
        #         raise ValueError('cluster should have at least one document to make sense')
        #     clu_vec = np.array([])
        #     for k_type in TokenSet.KEY_LIST:
        #         valid_ifd = valid_corpus_token_set.get(k_type)
        #         vocab_size = valid_ifd.vocabulary_size()
        #         type_tf_vec = np.zeros([vocab_size])
        #         clu_ifd = cluster.token_set.get(k_type)
        #         for word, freq in clu_ifd.word_freq_enumerate():
        #             type_tf_vec[valid_ifd.word2id(word)] = freq
        #         clu_vec = np.concatenate([clu_vec, type_tf_vec])
        #     cluid2vec[cluid] = clu_vec
        # """ make idf """
        # vec_len = sum([valid_corpus_token_set.get(k_type).vocabulary_size() for k_type in TokenSet.KEY_LIST])
        # print('vector length sum({})={}'.format(
        #     [valid_corpus_token_set.get(k_type).vocabulary_size() for k_type in TokenSet.KEY_LIST], vec_len))
        # d = len(cluid2vec)
        # for i in range(vec_len):
        #     df = 1
        #     for clu_vec in cluid2vec.values():
        #         if clu_vec[i] > 0:
        #             df += 1
        #     idf = np.log(d / df)
        #     for clu_vec in cluid2vec.values():
        #         clu_vec[i] *= idf
        # # tmu.check_time(print_func=lambda dt: print('construct tf-idf vector dt={}'.format(dt)))
        #
        # """ cosine similarity matrix """
        # cosine_matrix = au.cosine_matrix_multi([cluid2vec[cluid].reshape([-1]) for cluid in cluid_arr], process_num=16)
        # sim_matrix = pd.DataFrame(index=cluid_arr, columns=cluid_arr, data=0.0, dtype=np.float32)
        # for i in range(len(cluid_arr)):
        #     cluidi = cluid_arr[i]
        #     for j in range(i + 1, len(cluid_arr)):
        #         cluidj = cluid_arr[j]
        #         cos_sim = au.cosine_similarity(cluid2vec[cluidi], cluid2vec[cluidj])
        #         sim_matrix.loc[cluidi, cluidj] = sim_matrix.loc[cluidj, cluidi] = cos_sim
        # tmu.check_time(print_func=lambda dt: print('cosine similarity single dt={}'.format(dt)))
        
        # TODO for each cluster, get the four similarities with other clusters,
        # use them as features to make classification whether two clusters are of same label
        
        """ one matrix per type """
        type2vecarr = dict([(k_type, None) for k_type in TokenSet.KEY_LIST])
        cluid_arr = sorted(self.cludict.keys())
        valid_corpus_token_set = self.valid_corpus_token_set
        for k_type in TokenSet.KEY_LIST:
            valid_ifd = valid_corpus_token_set.get(k_type)
            vec_len = valid_ifd.vocabulary_size()
            for cluid in cluid_arr:
                clu_vec = np.zeros([vec_len])
                clu_ifd = self.cludict[cluid].token_set.get(k_type)
                for word, freq in clu_ifd.word_freq_enumerate():
                    clu_vec[valid_ifd.word2id(word)] = freq
                type2vecarr[k_type] = np.concatenate([type2vecarr[k_type], clu_vec.reshape([1, -1])]) \
                    if type2vecarr[k_type] is not None else clu_vec.reshape([1, -1])
            print(k_type, type2vecarr[k_type].shape)
        """ a matrix per type """
        w_dict = {su.pos_prop: 0.4, su.pos_comm: 0.3, su.pos_verb: 0.2, su.pos_hstg: 0.1}
        cosine_matrix = np.zeros([len(cluid_arr), len(cluid_arr)])
        for k_type in TokenSet.KEY_LIST:
            if 0 in type2vecarr[k_type].shape:
                continue
            cosmtx = au.cosine_similarity([vec.reshape([-1]) for vec in type2vecarr[k_type]],
                                          process_num=16)
            cosine_matrix += cosmtx * w_dict[k_type]
        
        """ ###  ### """
        """    ||    """
        """    __    """
        sim_matrix = pd.DataFrame(index=cluid_arr, columns=cluid_arr, data=cosine_matrix, dtype=np.float32)
        # tmu.check_time(print_func=lambda dt: print('cosine similarity multiple dt={}'.format(dt)))
        
        """ for each cluster, find top k similar clusters """
        top_k = 3
        cluid2topsim = dict()
        for cluid, row in sim_matrix.iterrows():
            top_sim_cluids = row.index[np.argsort(row.values)[::-1][:top_k]]
            cluid2topsim[cluid] = {'cluidarr': top_sim_cluids, 'scorearr': row[top_sim_cluids].tolist()}
        # tmu.check_time(print_func=lambda dt: print('find top 5 similar dt={}'.format(dt)))
        
        """ find representative label for every cluster """
        cluid2label = dict()
        rep_score = 0.7
        df = cs.cluid_label_table([int(i) for i in self.label], [int(i) for i in self.z])
        for cluid, row in df.iterrows():
            clu_twnum = sum(row.values)
            assert clu_twnum == self.cludict[cluid].twnum
            rep_label = int(row.index[np.argmax(row.values)])
            rep_twnum = row[rep_label]
            if rep_twnum == 0 or rep_twnum < clu_twnum * rep_score:
                cluid2label[cluid] = -1
            else:
                cluid2label[cluid] = rep_label
        # tmu.check_time(print_func=lambda dt: print('find representative label dt={}'.format(dt)))
        
        """ verify top sim. and rep. label """
        assert len(set(cluid2topsim.keys()).difference(set(cluid_arr))) == 0
        assert len(set(cluid2label.keys()).difference(set(cluid_arr))) == 0
        sim_info = list()
        for cluid in cluid_arr:
            clu_replb = cluid2label[cluid]
            clu_twnum = self.cludict[cluid].twnum
            sim_cluid_arr = cluid2topsim[cluid]['cluidarr']
            sim_score_arr = cluid2topsim[cluid]['scorearr']
            top_sim_cluid = sim_cluid_arr[0]
            top_sim_replb = cluid2label[top_sim_cluid]
            top_sim_twnum = self.cludict[top_sim_cluid].twnum
            top_sim_score = sim_score_arr[0]
            if top_sim_score < 0.4 or cluid >= top_sim_cluid:
                continue
            # print('\ncid {}, lb [{}], twnum {}'.format(cluid, clu_replb, clu_twnum))
            info = 'lb {:3} tw {:3}  <->  lb {:3} tw {:3}, score {}'.format(
                clu_replb, clu_twnum, top_sim_replb, top_sim_twnum, round(top_sim_score, 2))
            sim_info.append(info)
            # for idx in range(top_k):
            #     sim_cluid = sim_cluid_arr[idx]
            #     if sim_cluid <= cluid:
            #         continue
            #     sim_clu_twnum = self.cludict[sim_cluid].twnum
            #     print('    cid {:4}, lb [{:3}], score {}, twnum {}'.format(
            #         sim_cluid, cluid2label[sim_cluid], round(sim_score_arr[idx], 2), sim_clu_twnum))
        fu.dump_array(file_name, sim_info, False)
        # tmu.check_time(print_func=lambda dt: print('verify top sim. and rep. label dt={}'.format(dt)))
    
    def clusters_semantic_similarity(self, file_name):
        """ one matrix per type """
        cluid_arr = sorted(self.cludict.keys())
        clu_num = len(cluid_arr)
        type2cluvecs = dict([(k_type, []) for k_type in TokenSet.KEY_LIST])
        valid_corpus_token_set = self.valid_corpus_token_set
        for k_type in TokenSet.KEY_LIST:
            valid_ifd = valid_corpus_token_set.get(k_type)
            vec_len = valid_ifd.vocabulary_size()
            for cluid in cluid_arr:
                clu_vec = np.zeros([vec_len])
                clu_ifd = self.cludict[cluid].token_set.get(k_type)
                for word, freq in clu_ifd.word_freq_enumerate():
                    clu_vec[valid_ifd.word2id(word)] = freq
                type2cluvecs[k_type].append(clu_vec)
        """ one cos matrix per semantic class """
        # w_dict = {su.pos_prop: 0.4, su.pos_comm: 0.3, su.pos_verb: 0.2, su.pos_hstg: 0.1}
        # cosine_matrix = np.zeros([len(cluid_arr), len(cluid_arr)])
        type2cosmtx = dict()
        for k_type in TokenSet.KEY_LIST:
            vecmtx = np.array(type2cluvecs[k_type])
            if 0 in vecmtx.shape:
                cosmtx = np.zeros([clu_num, clu_num])
            else:
                cosmtx = au.cosine_similarity(vecmtx)
            print(cosmtx.shape)
            type2cosmtx[k_type] = cosmtx
        """ find representative label for every cluster """
        cluid2label = dict()
        rep_score = 0.7
        df = cs.cluid_label_table([int(i) for i in self.get_label()],
                                              [int(i) for i in self.get_z()])
        for cluid, row in df.iterrows():
            clu_twnum = sum(row.values)
            assert clu_twnum == self.cludict[cluid].twnum
            rep_label = int(row.index[np.argmax(row.values)])
            rep_twnum = row[rep_label]
            if rep_twnum == 0 or rep_twnum < clu_twnum * rep_score:
                cluid2label[cluid] = -1
                # cluster = self.cludict[cluid]
                # clu_counter = []
                # for label in row.index:
                #     if row[label] > 0:
                #         clu_counter.append((label, row[label]))
                # print('no representative label: {}'.format(clu_counter))
                # for twh in cluster.twhdict.values():
                #     print(twh[tk.key_text], twh.label)
                # print()
                # print()
            else:
                cluid2label[cluid] = rep_label
        """ use similarity as training data """
        for i in range(1, clu_num):
            cluid_i = cluid_arr[i]
            cluid_i_lb = cluid2label[cluid_i]
            for j in range(i + 1, clu_num):
                cluid_j = cluid_arr[j]
                cluid_j_lb = cluid2label[cluid_j]
                cos_sim_ij = [type2cosmtx[k_type][cluid_i][cluid_j] for k_type in TokenSet.KEY_LIST]
                print(cos_sim_ij,
                      au.cosine_similarity(type2cluvecs[k_type][cluid_i], type2cluvecs[k_type][cluid_j]))
                print()


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.twhdict = dict()
        self.token_set = TokenSet()
        self.twnum = 0
    
    def clear(self):
        self.twhdict.clear()
        self.token_set.clear()
        self.twnum = 0
    
    def update_by_twh(self, twh, factor):
        twh_token_set = twh.valid_token_set
        twh_id = twh.id
        if factor > 0:
            self.token_set.merge_freq_from(twh_token_set)
            self.twhdict[twh_id] = twh
            self.twnum += 1
        else:
            self.token_set.drop_freq_from(twh_token_set)
            self.twhdict.pop(twh_id)
            self.twnum -= 1


class TweetHolder:
    def __init__(self, tw):
        self.tw = tw
        self.id = tw.get(tk.key_id)
        self.label = None
        self.cluster = None
        self.orgn_token_set = TokenSet()
        self.valid_token_set = TokenSet()
        self.orgn_token_set.categorize(self.tw[tk.key_spacy])
    
    def __contains__(self, key):
        return key in self.tw
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.setdefault(key, value)
    
    def get(self, key):
        return self.tw.get(key, None)
    
    def setdefault(self, key, value):
        self.tw.setdefault(key, value)
    
    def get_cluid(self):
        return self.cluster.cluid
    
    def validate_tokenize(self, valid_corpus_token_set):
        self.valid_token_set.validate_from_tokensets(self.orgn_token_set, valid_corpus_token_set)
    
    def update_cluster(self, cluster):
        if self.cluster is not None:
            self.cluster.update_by_twh(self, factor=-1)
        self.cluster = cluster
        if cluster is not None:
            cluster.update_by_twh(self, factor=1)


class TokenSet:
    KEY_LIST = [su.pos_prop, su.pos_comm, su.pos_verb, su.pos_hstg]
    NORM_SET = {su.pos_prop, su.pos_comm, su.pos_verb}
    
    def __init__(self):
        self.type_ifd_dict = dict([(k_type, IdFreqDict()) for k_type in TokenSet.KEY_LIST])
    
    def get(self, k_type):
        return self.type_ifd_dict[k_type]
    
    def reset_id(self):
        for k_type in TokenSet.KEY_LIST:
            self.type_ifd_dict[k_type].reset_id()
    
    def clear(self):
        for ifd in self.type_ifd_dict.values():
            ifd.clear()
    
    def get_freq_sum(self, k_type):
        return self.type_ifd_dict[k_type].get_freq_sum()
    
    def merge_freq_from(self, token_set):
        for k_type in TokenSet.KEY_LIST:
            self.type_ifd_dict[k_type].merge_freq_from(token_set.type_ifd_dict[k_type])
    
    def drop_freq_from(self, token_set):
        for k_type in TokenSet.KEY_LIST:
            self.type_ifd_dict[k_type].drop_freq_from(token_set.type_ifd_dict[k_type])
    
    def drop_words_by_condition(self):
        for self_ifd in self.type_ifd_dict.values():
            self_ifd.drop_words_by_condition(3)
    
    def categorize(self, doc):
        for token in doc:
            word = token.text.strip().lower()
            token_tag = token.pos_
            if not pu.is_valid_keyword(word):
                continue
            if word.startswith('#'):
                self.type_ifd_dict[su.pos_hstg].count_word(word)
            elif token_tag in TokenSet.NORM_SET:
                self.type_ifd_dict[token_tag].count_word(word)
    
    def validate_from_tokensets(self, orgn_token_set, valid_token_set):
        # used for making validate dictionary according to the corpus valid_token_set
        for k_type, self_ifd in self.type_ifd_dict.items():
            self_ifd.clear()
            orgn_ifd = orgn_token_set.type_ifd_dict[k_type]
            valid_ifd = valid_token_set.type_ifd_dict[k_type]
            for word, freq in orgn_ifd.word_freq_enumerate(newest=False):
                if valid_ifd.has_word(word):
                    self_ifd.count_word(word, freq)

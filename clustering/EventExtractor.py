from collections import Counter

import numpy as np
import pandas as pd

import ArrayUtils
import TweetKeys
from Cache import CacheBack
from EventClassifier import LREventClassifier
from WordFreqCounter import WordFreqCounter


class EventExtractor:
    def __init__(self, dict_file, model_file):
        self.freqcounter = self.classifier = self.filteredtw = None
        self.construct(dict_file, model_file)
        self.cache_back = list()
        self.cache_front = list()
        self.inited = False
        self.tmp_list = list()
    
    def construct(self, dict_file, model_file):
        self.freqcounter = WordFreqCounter()
        self.freqcounter.load_worddict(dict_file)
        vocab_size = self.freqcounter.vocabulary_size()
        self.classifier = LREventClassifier(vocab_size=vocab_size, learning_rate=0,
                                            unlbreg_lambda=0.01, l2reg_lambda=0.01)
        self.classifier.load_params(model_file)
    
    def make_classification(self, twarr):
        feature_mtx = self.freqcounter.feature_matrix_of_twarr(twarr)
        return self.classifier.predict(feature_mtx)
    
    def filter_twarr(self, twarr, cond=lambda x: x >= 0.5):
        predicts = self.make_classification(twarr)
        return [twarr[idx] for idx, pred in enumerate(predicts) if cond(pred)]
    
    def merge_tw_into_cache_back(self, tw):
        if not self.inited:
            if not len(self.tmp_list) >= 15:
                self.tmp_list.append(tw)
            else:
                twarr_per_cache = ArrayUtils.array_partition(self.tmp_list, (1, 1, 1))
                for twarr in twarr_per_cache:
                    cache = CacheBack(self.freqcounter)
                    for tw in twarr:
                        cache.update_from_tw(tw)
                    self.cache_back.append(cache)
                self.inited = True
            return
        
        g_dict = dict()
        ng_dict = dict()
        k_dict = dict()
        for cache in self.cache_back:
            g_dict.update(cache.entities_geo.dictionary)
            ng_dict.update(cache.entities_non_geo.dictionary)
            k_dict.update(cache.keywords.dictionary)
        g_vocab = len(g_dict.keys())
        ng_vocab = len(ng_dict.keys())
        k_vocab = len(k_dict.keys())
        doc_num = sum([cache.tweet_number() for cache in self.cache_back])
        event_num = len(self.cache_back)
        alpha = 0.1
        beta = 0.1
        
        score_list = [cache.score_with_tw(tw, doc_num, event_num, g_vocab, ng_vocab, k_vocab, alpha, beta)
                      for cache in self.cache_back]
        max_score = np.max(score_list)
        max_score_idx = np.argmax(score_list)
        print(max_score_idx, '\t', max_score)
        
        if not max_score > 0.2:
            self.create_cache_with_tw(tw)
            return
        else:
            self.cache_back[max_score_idx].update_from_tw(tw)
            # print(tw[TweetKeys.key_cleantext])
            # print(score_list)
            # print('----\n')
            # self.cache_back[0].update_from_tw(tw)
    
    # def merge_tw_into_cache_back(self, tw):
    #     if not self.cache_back:
    #         self.create_cache_with_tw(tw)
    #         return
    #
    #     g_dict = dict()
    #     ng_dict = dict()
    #     k_dict = dict()
    #     for cache in self.cache_back:
    #         g_dict.update(cache.entities_geo.dictionary)
    #         ng_dict.update(cache.entities_non_geo.dictionary)
    #         k_dict.update(cache.keywords.dictionary)
    #     vocab = len(g_dict.keys()) + len(ng_dict.keys()) + len(k_dict.keys())
    #     doc_num = sum([cache.tweet_number() for cache in self.cache_back])
    #     event_num = len(self.cache_back)
    #     alpha = 0.1
    #     beta = 0.1
    #
    #     score_list = [cache.score_with_tw(tw, doc_num, event_num, vocab, alpha, beta)
    #                   for cache in self.cache_back]
    #     print(tw[TweetKeys.key_cleantext])
    #     print(score_list)
    #     print('----\n')
    #
    #     self.cache_back[0].update_from_tw(tw)
    
    def create_cache_with_tw(self, tw):
        cache = CacheBack(self.freqcounter)
        cache.update_from_tw(tw)
        self.cache_back.append(cache)
    
    def show_weight_of_words(self):
        thetaE = self.classifier.get_theta()[0]
        table = pd.DataFrame.from_dict(self.freqcounter.worddict.dictionary).T
        table.drop(axis=1, labels=['df'], inplace=True)
        table.sort_values(by='idf', ascending=False, inplace=True)
        for index, series in table.iterrows():
            table.loc[index, 'theta'] = thetaE[0][int(series['id'])]
        print(table)
    
    def GSDMM_twarr(self, twarr):
        twarr = twarr[:]
        words = dict()
        from Synset import get_root_word
        verb = {'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, }
        for tw in twarr:
            wordlabels = tw[TweetKeys.key_wordlabels]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                wordlabels[i][0] = get_root_word(wordlabels[i][0]) if wordlabels[i][2] in verb else wordlabels[i][0]
                if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            for wordlabel in wordlabels:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[TweetKeys.key_wordlabels]]))
    
        K = 15
        V = len(words.keys())
        D = len(twarr)
        iterNum = 50
        alpha = beta = 1e-1
        alpha0 = K * alpha
        beta0 = V * beta
        print('D', D, 'V', V)
    
        z = [0] * D
        m_z = [0] * K
        n_z = [0] * K
        n_zw = [[0] * V for _ in range(K)]
    
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            freq_dict = twarr[d]['dup']
            for word in freq_dict.keys():
                n_z[cluster] += freq_dict[word]
                n_zw[cluster][words[word]['id']] += freq_dict[word]
    
        """make sampling using current counting"""
        # smallDouble = 1e-150
        # largeDouble = 1e150
        # def recompute(prob, underflowcount):
        #     max_count = max(underflowcount)
        #     return [prob[k] * (largeDouble ** (underflowcount[k] - max_count)) for k in range(len(prob))]
        def sample_cluster(tw):
            prob = [0] * K
            # underflowcount = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                rule_value = 1.0
                i = 0
                freq_dict = tw['dup']
                for w, freq in freq_dict.items():
                    for j in range(freq):
                        wid = words[w]['id']
                        rule_value *= (n_zw[k][wid] + beta + j) / (n_z[k] + beta0 + i)
                        i += 1
                        # for i, wrdlbl in enumerate(tw[TweetKeys.key_wordlabels]):
                        # if rule_value < smallDouble:
                        #     underflowcount[k] -= 1
                        #     rule_value *= largeDouble
                prob[k] *= rule_value
            # prob = recompute(prob, underflowcount)
            return ArrayUtils.sample_index_by_array_value(np.array(prob))
    
        for i in range(iterNum):
            print(str(i) + '\t' + str(m_z) + '\n' if i % int(iterNum / 10) == 0 else '', end='')
            for d in range(D):
                cluster = z[d]
                m_z[cluster] -= 1
                freq_dict = twarr[d]['dup']
                for word in freq_dict.keys():
                    wordid = words[word]['id']
                    wordfreq = freq_dict[word]
                    n_zw[cluster][wordid] -= wordfreq
                    n_z[cluster] -= wordfreq
                cluster = sample_cluster(twarr[d])
                z[d] = cluster
                m_z[cluster] += 1
                for word in freq_dict.keys():
                    wordid = words[word]['id']
                    wordfreq = freq_dict[word]
                    n_zw[cluster][wordid] += wordfreq
                    n_z[cluster] += wordfreq
    
        tw_topic_arr = [[] for _ in range(K)]
        for d in range(D):
            tw_topic_arr[z[d]].append(twarr[d])
    
        for i, twarr in enumerate(tw_topic_arr):
            # if not len(twarr) > 6:
            #     continue
            # else:
            c = 1
            print('\n\ncluster', i, '\t\ttweet number', len(twarr))
            word_freq_list = [[word, n_zw[i][words[word]['id']]] for word in words.keys()]
            for pair in sorted(word_freq_list, key=lambda x: x[1], reverse=True)[:30]:
                print('{:<15}{:<5}'.format(pair[0], pair[1]), end='\n' if c % 5 == 0 else '\t')
                c += 1
        return tw_topic_arr
    
    
    
    def LECM_twarr(self, twarr):
        ner_pos_token = TweetKeys.key_wordlabels
        twarr = twarr[:]
        geo_words = dict()
        ent_words = dict()
        key_words = dict()
        
        def word_freq_id(word_dict, w):
            if w in word_dict:
                word_dict[w]['freq'] += 1
            else:
                word_dict[w] = {'freq': 1, 'id': len(word_dict.keys())}
        
        def word_count(word_dict, w):
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
        
        """for every tweet, count all its elements into the corresponding dictionary"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower()
                if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            # tw['geo'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if 'geo' in wlb[1]]))
            # tw['ent'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if
            #                           'geo' not in wlb[1] and not wlb[1].startswith('O')]))
            # tw['key'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[1].startswith('O')]))
            tw['geo'] = dict()
            tw['ent'] = dict()
            tw['key'] = dict()
            
            for wordlabel in wordlabels:
                word = wordlabel[0]
                ner = wordlabel[1]
                if 'geo' in ner:
                    word_freq_id(geo_words, word)
                    word_count(tw['geo'], word)
                elif not ner.startswith('O'):
                    word_freq_id(ent_words, word)
                    word_count(tw['ent'], word)
                else:
                    word_freq_id(key_words, word)
                    word_count(tw['key'], word)
        
        iterNum = 50
        
        K = 10
        D = len(twarr)
        L = len(geo_words.keys())
        Y = len(ent_words.keys())
        V = len(key_words.keys())
        alpha = beta = eta = lambd = 0.1
        alpha0 = alpha * K
        beta0 = beta * Y
        eta0 = eta * L
        lambd0 = lambd * V
        
        z = [0] * D
        m_z = [0] * K
        n_z_geo = [0] * K
        n_z_ent = [0] * K
        n_z_key = [0] * K
        n_zw_geo = [[0] * L for _ in range(K)]
        n_zw_ent = [[0] * Y for _ in range(K)]
        n_zw_key = [[0] * V for _ in range(K)]
        
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            tw_geo_freq_dict = twarr[d]['geo']
            tw_ent_freq_dict = twarr[d]['ent']
            tw_key_freq_dict = twarr[d]['key']
            for word in tw_geo_freq_dict.keys():
                n_z_geo[cluster] += tw_geo_freq_dict[word]
                n_zw_geo[cluster][geo_words[word]['id']] += tw_geo_freq_dict[word]
            for word in tw_ent_freq_dict.keys():
                n_z_ent[cluster] += tw_ent_freq_dict[word]
                n_zw_ent[cluster][ent_words[word]['id']] += tw_ent_freq_dict[word]
            for word in tw_key_freq_dict.keys():
                n_z_key[cluster] += tw_key_freq_dict[word]
                n_zw_key[cluster][key_words[word]['id']] += tw_key_freq_dict[word]
        
        """make sampling using current counting"""
        def sample_cluster(tw):
            # wrdlbl = tw[ner_pos_token]
            geo_freq_dict = tw['geo']
            ent_freq_dict = tw['ent']
            key_freq_dict = tw['key']
            prob = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D + alpha0)
                
                b = 1
                rule_value = 1.0
                for geo_w, w_count in geo_freq_dict.items():
                    for idx in range(1, w_count + 1):
                        wid = geo_words[geo_w]['id']
                        rule_value *= (n_zw_geo[k][wid] + idx + eta)/(n_z_geo[k] + b + eta0)
                        b += 1
                prob[k] *= rule_value
                b = 1
                rule_value = 1.0
                for ent_w, w_count in ent_freq_dict.items():
                    for idx in range(1, w_count + 1):
                        wid = ent_words[ent_w]['id']
                        rule_value *= (n_zw_ent[k][wid] + idx + beta)/(n_z_ent[k] + b + beta0)
                        b += 1
                b = 1
                prob[k] *= rule_value
                rule_value = 1.0
                for key_w, w_count in key_freq_dict.items():
                    for idx in range(1, w_count + 1):
                        wid = key_words[key_w]['id']
                        rule_value *= (n_zw_key[k][wid] + idx + lambd)/(n_z_key[k] + b + lambd0)
                        b += 1
                prob[k] *= rule_value
                # if rule_value < smallDouble:
                #     underflowcount[k] -= 1
                #     rule_value *= largeDouble
                # prob = recompute(prob, underflowcount)
            return ArrayUtils.sample_index_by_array_value(np.array(prob))
        
        def update_using_freq_dict(tw_freq_dict, wordid_dict, n_z_, n_zw_, factor):
            for w, w_freq in tw_freq_dict.items():
                w_id = wordid_dict[w]['id']
                w_freq *= factor
                n_z_[cluster] += w_freq
                n_zw_[cluster][w_id] += w_freq
        
        for i in range(iterNum):
            print(str(i) + '\t' + str(m_z) + '\n' if i % int(iterNum / 10) == 0 else '', end='')
            for d in range(D):
                tw = twarr[d]
                tw_geo_freq_dict = tw['geo']
                tw_ent_freq_dict = tw['ent']
                tw_key_freq_dict = tw['key']
                
                cluster = z[d]
                m_z[cluster] -= 1
                # for geo_w, geo_w_freq in tw_geo_freq_dict.items():
                #     geo_w_id = geo_words[geo_w]['id']
                #     n_z_geo[cluster] -= geo_w_freq
                #     n_zw_geo[cluster][geo_w_id] -= geo_w_freq
                # for ent_w, ent_w_freq in tw_ent_freq_dict.items():
                #     ent_w_id = ent_words[ent_w]['id']
                #     n_z_ent[cluster] -= ent_w_freq
                #     n_zw_ent[cluster][ent_w_id] -= ent_w_freq
                # for key_w, key_w_freq in tw_key_freq_dict.items():
                #     key_w_id = key_words[key_w]['id']
                #     n_z_key[cluster] -= key_w_freq
                #     n_zw_key[cluster][key_w_id] -= key_w_freq
                update_using_freq_dict(tw_geo_freq_dict, geo_words, n_z_geo, n_zw_geo, 1)
                update_using_freq_dict(tw_ent_freq_dict, ent_words, n_z_ent, n_zw_ent, 1)
                update_using_freq_dict(tw_key_freq_dict, key_words, n_z_key, n_zw_key, 1)
                
                cluster = sample_cluster(tw)
                
                z[d] = cluster
                m_z[cluster] += 1
                update_using_freq_dict(tw_geo_freq_dict, geo_words, n_z_geo, n_zw_geo, -1)
                update_using_freq_dict(tw_ent_freq_dict, ent_words, n_z_ent, n_zw_ent, -1)
                update_using_freq_dict(tw_key_freq_dict, key_words, n_z_key, n_zw_key, -1)

            tw_topic_arr = [[] for _ in range(K)]
            for d in range(D):
                tw_topic_arr[z[d]].append(twarr[d])
            
            # for i, twarr in enumerate(tw_topic_arr):
            #     if not len(twarr) > 6:
            #         continue
            #     else:
            #         c = 1
            #         print('\n\ncluster', i)
            #         word_freq_list = [[word, n_zw[i][words[word]['id']]] for word in words.keys()]
            #         for pair in sorted(word_freq_list, key=lambda x: x[1], reverse=True)[:30]:
            #             print('{:<15}{:<5}'.format(pair[0], pair[1]), end='\n' if c % 5 == 0 else '\t')
            #             c += 1
            return tw_topic_arr

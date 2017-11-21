from collections import Counter

from EventClassifier import EventClassifier
from WordFreqCounter import WordFreqCounter
from Cache import CacheBack, CacheFront
import ArrayUtils
import TweetKeys

import pandas as pd
import numpy as np


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
        self.classifier = EventClassifier(vocab_size=vocab_size, learning_rate=0)
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
        """init"""
        for tw in twarr:
            wordlabels = tw[TweetKeys.key_wordlabels]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower()
                if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            for wordlabel in wordlabels:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[TweetKeys.key_wordlabels]]))
        # for idx, word in enumerate(sorted(words.keys())):
        #     words[word]['id'] = idx
        
        K = 6
        V = len(words.keys())
        D = len(twarr)
        iterNum = 100
        alpha = beta = 1e-1
        alpha0 = K * alpha
        beta0 = V * beta
        print('D', D, 'V', V)
        
        z = [0] * D
        m_z = [0] * K
        n_z = [0] * K
        n_zw = [[0] * V for _ in range(K)]
        
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            freq_dict = twarr[d]['dup']
            for word in freq_dict.keys():
                n_z[cluster] += freq_dict[word]
                n_zw[cluster][words[word]['id']] += freq_dict[word]
        
        # smallDouble = 1e-150
        # largeDouble = 1e150
        # def recompute(prob, underflowcount):
        #     max_count = max(underflowcount)
        #     return [prob[k] * (largeDouble ** (underflowcount[k] - max_count)) for k in range(len(prob))]
        def sampleCluster(tw):
            prob = [0] * K
            # underflowcount = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                rule_value = 1.0
                for i, wrdlbl in enumerate(tw[TweetKeys.key_wordlabels]):
                    # if rule_value < smallDouble:
                    #     underflowcount[k] -= 1
                    #     rule_value *= largeDouble
                    wid = words[wrdlbl[0]]['id']
                    rule_value *= (n_zw[k][wid] + beta) / (n_z[k] + beta0 + i)
                prob[k] *= rule_value
            # prob = recompute(prob, underflowcount)
            # return ArrayUtils.sample_index_by_array_value(prob)
            return ArrayUtils.sample_index_by_array_value(np.array(prob) * 1e30)
        
        for i in range(iterNum):
            "尝试不同的遍历方式，不一定是每次遍历所有推文"
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
                cluster = sampleCluster(twarr[d])
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
        
        # print(sum([wordattr['freq'] for wordattr in words.values()]))
        
        for i, twarr in enumerate(tw_topic_arr):
            if not len(twarr) > 6:
                continue
            else:
                c = 1
                print('\n\ncluster', i)
                word_freq_list = [[word, n_zw[i][words[word]['id']]] for word in words.keys()]
                for pair in sorted(word_freq_list, key=lambda x: x[1], reverse=True)[:30]:
                    print('{:<15}{:<5}'.format(pair[0], pair[1]), end='\n' if c % 5 == 0 else '\t')
                    c += 1
        return tw_topic_arr

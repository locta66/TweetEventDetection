import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Configure import getconfig
import TweetKeys
import ArrayUtils as au
import FunctionUtils as fu
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
                twarr_per_cache = au.array_partition(self.tmp_list, (1, 1, 1))
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
    
    def cluster_label_prediction_table(self, tw_cluster_label, tw_cluster_pred):
        c_distribution = pd.DataFrame(index=set(tw_cluster_pred), columns=set(tw_cluster_label), data=0)
        for i in range(len(tw_cluster_label)):
            c_distribution.loc[tw_cluster_pred[i]][tw_cluster_label[i]] += 1
        return c_distribution
    
    def print_cluster_prediction_table(self, tw_cluster_label, tw_cluster_pred):
        print(self.cluster_label_prediction_table(tw_cluster_label, tw_cluster_pred))
    
    def event_recall(self, tw_cluster_label, tw_cluster_pred):
        c_distribution = self.cluster_label_prediction_table(tw_cluster_label, tw_cluster_pred)
        predict_cluster = set(np.argmax(c_distribution.values, axis=1))
        ground_cluster = set(tw_cluster_label)
        recall = len(predict_cluster) / len(ground_cluster)
        return c_distribution, recall, predict_cluster, ground_cluster
    
    def create_clusters_with_labels(self, twarr, tw_cluster_label):
        if not len(twarr) == len(tw_cluster_label):
            raise ValueError('Wrong cluster labels for twarr')
        # tw_cluster_label can be either prediction or ground-truth label for every twitter
        tw_topic_arr = [[] for _ in range(max(tw_cluster_label) + 1)]
        for d in range(len(tw_cluster_label)):
            tw_topic_arr[tw_cluster_label[d]].append(twarr[d])
        return tw_topic_arr
    
    def GSDMM_twarr_with_label(self, twarr, tw_cluster_label):
        # line_style = ['-', '^-', '+-', 'x-', '<-', '-.']
        line_color = ['#FF0000', '#FFAA00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF',
                      '#BFBFBF', '#A020F0', '#8B5A00', '#C71585', '#87CEFF', '#4B0082', '']
        alpha_range = beta_range = [i/100 for i in range(1, 10, 3)] + [i/10 for i in range(1, 10, 3)] + \
                                   [i for i in range(1, 10, 3)]
        K_range = [20, 30, 40, 50]
        # alpha_range = beta_range = [0.1, 0.2]
        # alpha_range = beta_range = [0.01, 0.1, 0.5, 1, 1.5, 2]
        # topic_word_dstrb = tw_cluster_pred = nmi = 0
    
        """cluster using different hyperparams in multiprocess way"""
        iter_num = 100
        process_num = 18
        tempobj = TempObject(self.freqcounter)
        hyperparams = [(a, b, K) for a in alpha_range for b in beta_range for K in K_range]
        res_list = list()
        for i in range(int(math.ceil(len(hyperparams) / process_num))):
            param_list = [(tempobj, twarr, *param, iter_num, tw_cluster_label) for param in
                          hyperparams[i * process_num: (i + 1) * process_num]]
            res_list += fu.multi_process(EventExtractor.GSDMM_twarr, param_list)
            print('{:<4} /'.format((i + 1) * process_num), len(hyperparams), 'params processed')
        """group the data by alpha and K"""
        frame = pd.DataFrame(index=np.arange(0, len(hyperparams)), columns=['alpha', 'beta', 'K'])
        for i in range(len(hyperparams)):
            frame.loc[i] = hyperparams[i]
        print('\n', frame, '\n')
        """start plotting figures"""
        for (alpha, K), indices in frame.groupby(['alpha', 'K']).groups.items():
            color = line_color[:]
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(8)
            for i in indices:
                beta = frame.loc[i]['beta']
                topic_word_dstrb, tw_cluster_pred, iter_x, nmi_y, homo_y, cmplt_y = res_list[i]
                # print(alpha, beta, K, ':', np.array(indices))
                plt.plot(iter_x, nmi_y, '-', lw=1.5, color=color.pop(0), label='beta=' + str(round(beta, 2)))
            plt.xlabel('iteration')
            plt.ylabel('NMI')
            plt.ylim(0.25, 0.75)
            plt.title('alpha=' + str(round(alpha, 2)) + ',K=' + str(K))
            plt.legend(loc='lower right')
            plt.grid(True, '-', color='#333333', lw=0.8)
            plt.text(iter_num - 20, 0.70, 'final nmi: ' + str(round(max([res_list[i][3][-1] for i in indices]), 6)),
                     fontsize=15, verticalalignment='bottom', horizontalalignment='left')
            plt.savefig(getconfig().dc_test + 'alpha=' + str(round(alpha, 2)) + '_K=' + str(K) + '.png')
        
        # for alpha in alpha_range:
        #     color = line_color[:]
        #     fig = plt.figure()
        #     fig.set_figheight(8)
        #     fig.set_figwidth(8)
        #     for beta in beta_range:
        #         topic_word_dstrb, tw_cluster_pred_, iter_x, nmi_y, homo_y, cmplt_y = \
        #             self.GSDMM_twarr(twarr, alpha, beta, 30, 60, tw_cluster_label)
        #         plt.plot(iter_x, nmi_y, '-', lw=1, color=color.pop(0), label='beta=' + str(round(beta, 2)))
        #         nmi_ = au.score(tw_cluster_label, tw_cluster_pred_, score_type='nmi')
        #         print('alpha {:<5}, beta {:<5}, NMI {:<8}'. format(alpha, beta, nmi_))
        #         if nmi < nmi_:
        #             tw_cluster_pred, nmi = tw_cluster_pred_, nmi_
        #
        #     plt.xlabel('iteration')
        #     plt.ylim(0.25, 0.75)
        #     plt.ylabel('NMI')
        #     plt.title('alpha=' + str(round(alpha, 2)) + ',K=' + str(30))
        #     plt.legend(loc='lower right')
        #     plt.grid(True, '-', color="#333333", lw=0.8)
        #     plt.savefig('alpha_' + str(round(alpha, 2)) + '.png')
        #
        # # nmi_frame = pd.DataFrame(index=alpha_range, columns=beta_range, data=0.0)
        # # nmi_frame.loc[alpha][beta] = nmi_
        # # ax = Axes3D(fig)
        # # X, Y = np.meshgrid(alpha_range, beta_range)
        # # ax.plot_surface(X, Y, np.array(nmi_frame.T).tolist(), cmap='rainbow')
        # # fig.savefig('ohno.png')
        #
        # print('max NMI:', nmi)
        # self.cluster_to_label(tw_cluster_label, tw_cluster_pred)
        #
        # tw_topic_arr = [[] for _ in range(len(topic_word_dstrb))]
        # for d in range(len(twarr)):
        #     tw_topic_arr[tw_cluster_pred[d]].append(twarr[d])
        # return tw_topic_arr, tw_cluster_pred
        return None, None
    
    def GSDMM_twarr(self, twarr, alpha, beta, K, iter_num, ref_labels=None):
        ner_pos_token = TweetKeys.key_wordlabels
        twarr = twarr[:]
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        # from Synset import get_root_word
        # verb = {'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, }
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                # wordlabels[i][0] = get_root_word(wordlabels[i][0]) if wordlabels[i][2] in verb else \
                #     wordlabels[i][0]
                if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            for wordlabel in wordlabels:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
        min_df = 3
        for w in list(words.keys()):
            if words[w]['freq'] < min_df:
                del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[0] in words]))
        """definitions of parameters"""
        V = len(words.keys())
        D = len(twarr)
        alpha0 = K * alpha
        beta0 = V * beta
        # print('D', D, 'V', V)
        
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
        """make sampling using current counting information"""
        small_double = 1e-150
        large_double = 1e150
        def recompute(prob, underflowcount):
            max_count = max(underflowcount)
            return [prob[k] * (large_double ** (underflowcount[k] - max_count)) for k in range(len(prob))]
        def sample_cluster(tw, iter=None):
            prob = [0] * K
            underflowcount = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                rule_value = 1.0
                i = 0
                freq_dict = tw['dup']
                for w, freq in freq_dict.items():
                    for j in range(freq):
                        wid = words[w]['id']
                        rule_value *= (n_zw[k][wid] + beta + j) / (n_z[k] + beta0 + i)
                        if rule_value < small_double:
                            underflowcount[k] -= 1
                            rule_value *= large_double
                        i += 1
                prob[k] *= rule_value
            
            prob = recompute(prob, underflowcount)
            if iter is not None and iter > 90:
                return np.argmax(prob)
            else:
                return au.sample_index_by_array_value(np.array(prob))
        
        """start iteration"""
        iter_x = list()
        nmi_y = list()
        homo_y = list()
        cmplt_y = list()
        for i in range(iter_num):
            for d in range(D):
                cluster = z[d]
                m_z[cluster] -= 1
                freq_dict = twarr[d]['dup']
                for word in freq_dict.keys():
                    wordid = words[word]['id']
                    wordfreq = freq_dict[word]
                    n_zw[cluster][wordid] -= wordfreq
                    n_z[cluster] -= wordfreq
                
                cluster = sample_cluster(twarr[d], i)
                
                z[d] = cluster
                m_z[cluster] += 1
                for word in freq_dict.keys():
                    wordid = words[word]['id']
                    wordfreq = freq_dict[word]
                    n_zw[cluster][wordid] += wordfreq
                    n_z[cluster] += wordfreq
            
            # if i % int((iter_num + 1) / 10) == 0:
            #     print(str(i) + '\t' + str(m_z))
            if ref_labels is not None:
                iter_x.append(i)
                nmi_y.append(au.score(z, ref_labels, score_type='nmi'))
                homo_y.append(au.score(z, ref_labels, score_type='homo'))
                cmplt_y.append(au.score(z, ref_labels, score_type='cmplt'))
        
        # """make clusters according to the labels"""
        # tw_topic_arr = [[] for _ in range(K)]
        # for d in range(D):
        #     tw_topic_arr[z[d]].append(twarr[d])
        
        # """make conclusion"""
        # for i, twarr in enumerate(tw_topic_arr):
        #     if not len(twarr) == 0:
        #         c = 1
        #         print('\n\ncluster', i, '\t\ttweet number', len(twarr))
        #         word_freq_list = [[word, n_zw[i][words[word]['id']]] for word in words.keys()]
        #         for pair in sorted(word_freq_list, key=lambda x: x[1], reverse=True)[:30]:
        #             print('{:<15}{:<5}'.format(pair[0], pair[1]), end='\n' if c % 5 == 0 else '\t')
        #             c += 1
        if ref_labels is not None:
            return n_zw, z, iter_x, nmi_y, homo_y, cmplt_y
        else:
            return n_zw, z
    
    def GSDPMM_twarr_with_label(self, twarr, tw_cluster_label):
        line_color = ['#FF0000', '#FFAA00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF',
                      '#BFBFBF', '#A020F0', '#8B5A00', '#C71585', '#87CEFF', '#4B0082', ]
        alpha_range = beta_range = [i / 100 for i in range(1, 10, 2)] + [i / 10 for i in range(1, 10, 2)]
        """cluster using different hyperparams in multiprocess way"""
        iter_num = 120
        process_num = 19
        tempobj = TempObject(self.freqcounter)
        hyperparams = [(a, b) for a in alpha_range for b in beta_range]
        res_list = list()
        for i in range(int(math.ceil(len(hyperparams) / process_num))):
            param_list = [(tempobj, twarr, *param, iter_num, tw_cluster_label) for param in
                          hyperparams[i * process_num: (i + 1) * process_num]]
            res_list += fu.multi_process(EventExtractor.GSDPMM_twarr, param_list)
            print('{:<3} /'.format(min((i + 1) * process_num, len(hyperparams))), len(hyperparams), 'params processed')
        """group the data by alpha"""
        frame = pd.DataFrame(index=np.arange(0, len(hyperparams)), columns=['alpha', 'beta'])
        for i in range(len(hyperparams)):
            frame.loc[i] = hyperparams[i]
        """start plotting figures"""
        for alpha, indices in frame.groupby('alpha').groups.items():
            color = line_color[:]
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(8)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            for i in indices:
                beta = frame.loc[i]['beta']
                topic_word_dstrb, tw_cluster_pred, iter_x, nmi_y, k_y = res_list[i]
                # print(alpha, beta, ':', np.array(indices))
                l_color = color.pop(0)
                ax1.plot(iter_x, nmi_y, '-', lw=1.5, color=l_color, label='beta=' + str(round(beta, 2)))
                ax2.plot(iter_x, k_y, '^', lw=1.5, color=l_color, label='beta=' + str(round(beta, 2)))
            ax1.set_title('alpha=' + str(round(alpha, 2)))
            ax1.set_ylabel('NMI')
            ax1.set_ylim(0.25, 0.75)
            ax1.legend(loc='lower right')
            ax1.text(iter_num * 0.6, 0.70,
                     'final nmi: ' + str(round(max([res_list[i][3][-1] for i in indices]), 4)), fontsize=15)
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('K num')
            ax2.legend(loc='lower right')
            plt.grid(True, '-', color='#333333', lw=0.8)
            plt.savefig(getconfig().dc_test + 'GSDPMM_alpha=' + str(round(alpha, 2)) + '.png')
        
        top_K = 10
        alpha_idx = 0
        beta_idx = 1
        tw_cluster_pred_idx = 3
        nmi_idx = 5
        # k_idx = 6
        # c_distr_idx = 7
        recall_idx = 8
        # pred_cluster_set_idx = 9
        
        summary_list = [hyperparams[i] + res_list[i] + self.event_recall(tw_cluster_label, res_list[i][1])
                        for i in range(len(hyperparams))]
        top_recall_summary_list = [summary_list[i] for i in
                                   np.argsort([summary[recall_idx] for summary in summary_list])[::-1][:top_K]]
        top_nmi_summary_list = [summary_list[i] for i in
                                np.argsort([summary[nmi_idx][-1] for summary in summary_list])[::-1][:top_K]]
        
        import os
        import TweetKeys
        top_nmi_path = '/home/nfs/cdong/tw/testdata/cdong/max_nmis/'
        top_recall_path = '/home/nfs/cdong/tw/testdata/cdong/max_recalls/'
        fu.rmtree(top_nmi_path)
        fu.rmtree(top_recall_path)
        
        def dump_cluster_info(summary_list, base_path):
            for rank, summary in enumerate(summary_list):
                res_dir = base_path + '{}_recall_{}_nmi_{}_alpha_{}_beta_{}/'.\
                    format(rank, round(summary[recall_idx], 6), round(summary[nmi_idx][-1], 6),
                           summary[alpha_idx], summary[beta_idx])
                os.makedirs(res_dir)
                tw_topic_arr = self.create_clusters_with_labels(twarr, summary[tw_cluster_pred_idx])
                for i, _twarr in enumerate(tw_topic_arr):
                    if not len(_twarr) == 0:
                        fu.dump_array(res_dir + str(i) + '.txt', [tw[TweetKeys.key_cleantext] for tw in _twarr])
                table = self.cluster_label_prediction_table(tw_cluster_label, summary[tw_cluster_pred_idx])
                table.to_csv(res_dir + 'table.csv')
        
        dump_cluster_info(top_recall_summary_list, top_recall_path)
        dump_cluster_info(top_nmi_summary_list, top_nmi_path)
        
        return None, None
    
    def GSDPMM_twarr(self, twarr, alpha, beta, iter_num, ref_labels=None):
        ner_pos_token = TweetKeys.key_wordlabels
        twarr = twarr[:]
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            for wordlabel in wordlabels:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
        min_df = 4
        for w in list(words.keys()):
            if words[w]['freq'] < min_df:
                del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[0] in words]))
        """definitions of parameters"""
        K = 1       # default 1 set by the algorithm
        D = len(twarr)
        V = len(words.keys())
        beta0 = V * beta
        z = [0] * D
        m_z = [0] * K
        n_z = [0] * K
        n_zw = [[0] * V for _ in range(K)]
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            for word, freq in twarr[d]['dup'].items():
                n_z[cluster] += freq
                n_zw[cluster][words[word]['id']] += freq
        """make sampling using current counting information"""
        def sample_cluster(tw, iter=None):
            prob = [0] * K
            freq_dict = tw['dup']
            for k in range(K):
                prob[k] = m_z[k] / (D - 1 + alpha)
                rule_value = 1.0
                i_ = 0
                for word, freq in freq_dict.items():
                    for j_ in range(freq):
                        rule_value *= (n_zw[k][words[word]['id']] + beta + j_) / (n_z[k] + beta0 + i_)
                        i_ += 1
                prob[k] *= rule_value
            new_cluster_prob = alpha / (D - 1 + alpha)
            i_ = 0
            for word, freq in freq_dict.items():
                for j_ in range(freq):
                    new_cluster_prob *= (beta + j_) / (beta0 + i_)
                    i_ += 1
            if iter is not None and iter > iter_num - 5:
                return np.argmax(prob + [new_cluster_prob])
            else:
                return au.sample_index_by_array_value(np.array(prob + [new_cluster_prob]))
        """start iteration"""
        iter_x = list()
        nmi_y = list()
        k_y = list()
        for i in range(iter_num):
            for d in range(D):
                freq_dict = twarr[d]['dup']
                cluster = z[d]
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
                    for d_ in range(D):
                        if z[d_] > min_tw_num_idx:
                            z[d_] -= 1
                
                cluster = sample_cluster(twarr[d], i)
                
                if cluster >= K:
                    m_z.append(0)
                    n_z.append(0)
                    n_zw.append([0] * V)
                    K += 1
                
                z[d] = cluster
                m_z[cluster] += 1
                for word, freq in freq_dict.items():
                    n_z[cluster] += freq
                    n_zw[cluster][words[word]['id']] += freq
            
            if ref_labels is not None:
                iter_x.append(i)
                nmi_y.append(au.score(z, ref_labels, score_type='nmi'))
                k_y.append(K)
        
        if ref_labels is not None:
            return n_zw, z, iter_x, nmi_y, k_y
        else:
            return n_zw, z

    # def LECM_twarr_with_label(self, twarr, tw_cluster_label):
    #     # Currently best hyperparam 1, 0.1, 0.1, 1
    #     tw_topic_arr, tw_cluster_pred = self.LECM_twarr(twarr, 1, 0.1, 0.1, 1, 20, 1)
    #     print('one epoch:alpha {:<5}, eta {:<5}, beta {:<5}, lambd {:<5}, NMI {:<8}\n'.
    #           format(0.1, 0.1, 0.1, 0.1, normalized_mutual_info_score(tw_cluster_pred, tw_cluster_label)))
    #     tw_topic_arr = tw_cluster_pred = nmi = 0
    #     for alpha in [1]:
    #         for eta in [0.1]:
    #             for beta in [0.1]:
    #                 for lambd in [1]:
    #                     tw_topic_arr_, tw_cluster_pred_ = self.LECM_twarr(twarr, alpha, eta, beta, lambd, 20, 70)
    #                     nmi_ = au.nmi_score(tw_cluster_pred_, tw_cluster_label)
    #                     print('alpha {:<5}, eta {:<5}, beta {:<5}, lambd {:<5}, NMI{:<8}'.
    #                           format(alpha, eta, beta, lambd, nmi_))
    #                     if nmi < nmi_:
    #                         tw_topic_arr, tw_cluster_pred = tw_topic_arr_, tw_cluster_pred_
    #                         nmi = nmi_
    #     return tw_topic_arr, tw_cluster_pred
    #
    # def LECM_twarr(self, twarr, alpha, eta, beta, lambd, K, iter_num):
    #     ner_pos_token = TweetKeys.key_wordlabels
    #     twarr = twarr[:]
    #     geo_word_id_dict = dict()
    #     ent_word_id_dict = dict()
    #     key_word_id_dict = dict()
    #
    #     def word_count_id(word_dict, w):
    #         if w in word_dict:
    #             word_dict[w]['freq'] += 1
    #         else:
    #             word_dict[w] = {'freq': 1, 'id': word_dict.__len__()}
    #
    #     def word_count_freq(word_dict, w):
    #         if w in word_dict:
    #             word_dict[w] += 1
    #         else:
    #             word_dict[w] = 1
    #
    #     """for every tweet, count all its elements into the corresponding dictionary"""
    #     for tw in twarr:
    #         wordlabels = tw[ner_pos_token]
    #         for i in range(len(wordlabels) - 1, -1, -1):
    #             wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
    #             if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
    #                 del wordlabels[i]
    #         tw['geo'] = dict()
    #         tw['ent'] = dict()
    #         tw['key'] = dict()
    #         for wordlabel in wordlabels:
    #             word = wordlabel[0]
    #             ner = wordlabel[1]
    #             if 'geo' in ner:
    #                 word_count_id(geo_word_id_dict, word)
    #                 word_count_freq(tw['geo'], word)
    #             elif not ner.startswith('O'):
    #                 word_count_id(ent_word_id_dict, word)
    #                 word_count_freq(tw['ent'], word)
    #             else:
    #                 word_count_id(key_word_id_dict, word)
    #                 word_count_freq(tw['key'], word)
    #
    #     # iterNum = 70
    #     # K = 40
    #     # alpha = 0.1
    #     # eta = 0.1
    #     # beta = 0.5
    #     # lambd = 0.3
    #     """cluster level"""
    #     D = twarr.__len__()
    #     alpha0 = alpha * K
    #     """geo level"""
    #     L = geo_word_id_dict.__len__()
    #     eta0 = eta * L
    #     """non geo level"""
    #     Y = ent_word_id_dict.__len__()
    #     beta0 = beta * Y
    #     """keyword level"""
    #     V = key_word_id_dict.__len__()
    #     lambd0 = lambd * V
    #
    #     print('D', D, 'L', L, 'Y', Y, 'V', V)
    #     # print('alpha', alpha, 'eta', eta, 'beta', beta, 'lambd', lambd)
    #
    #     z = [0] * D
    #     m_z = [0] * K
    #     n_z_geo = [0] * K
    #     n_z_ent = [0] * K
    #     n_z_key = [0] * K
    #     n_zw_geo = [[0] * L for _ in range(K)]
    #     n_zw_ent = [[0] * Y for _ in range(K)]
    #     n_zw_key = [[0] * V for _ in range(K)]
    #
    #     """initialize the counting arrays"""
    #     for d in range(D):
    #         cluster = int(K * np.random.random())
    #         z[d] = cluster
    #         m_z[cluster] += 1
    #         tw_geo_freq_dict = twarr[d]['geo']
    #         tw_ent_freq_dict = twarr[d]['ent']
    #         tw_key_freq_dict = twarr[d]['key']
    #         for word in tw_geo_freq_dict.keys():
    #             n_z_geo[cluster] += tw_geo_freq_dict[word]
    #             n_zw_geo[cluster][geo_word_id_dict[word]['id']] += tw_geo_freq_dict[word]
    #         for word in tw_ent_freq_dict.keys():
    #             n_z_ent[cluster] += tw_ent_freq_dict[word]
    #             n_zw_ent[cluster][ent_word_id_dict[word]['id']] += tw_ent_freq_dict[word]
    #         for word in tw_key_freq_dict.keys():
    #             n_z_key[cluster] += tw_key_freq_dict[word]
    #             n_zw_key[cluster][key_word_id_dict[word]['id']] += tw_key_freq_dict[word]
    #
    #     """make sampling using current counting"""
    #     def rule_value_of(tw_freq_dict_, word_id_dict, n_z_, n_zw_, p, p0, cluster):
    #         b = 1
    #         value = 1.0
    #         for w_, w_count in tw_freq_dict_.items():
    #             for idx in range(1, w_count + 1):
    #                 wid = word_id_dict[w_]['id']
    #                 value *= (n_zw_[cluster][wid] + idx + p) / (n_z_[cluster] + b + p0)
    #                 b += 1
    #         return value
    #
    #     def sample_cluster(tw):
    #         geo_freq_dict = tw['geo']
    #         ent_freq_dict = tw['ent']
    #         key_freq_dict = tw['key']
    #         prob = [0] * K
    #         for k in range(K):
    #             prob[k] = (m_z[k] + alpha) / (D + alpha0)
    #             prob[k] *= rule_value_of(geo_freq_dict, geo_word_id_dict, n_z_geo, n_zw_geo, eta, eta0, k)
    #             prob[k] *= rule_value_of(ent_freq_dict, ent_word_id_dict, n_z_ent, n_zw_ent, beta, beta0, k)
    #             prob[k] *= rule_value_of(key_freq_dict, key_word_id_dict, n_z_key, n_zw_key, lambd, lambd0, k)
    #             # bb=1.0
    #             # b = 1
    #             # rule_value = 1.0
    #             # for geo_w, w_count in geo_freq_dict.items():
    #             #     for idx in range(1, w_count + 1):
    #             #         wid = geo_word_id_dict[geo_w]['id']
    #             #         rule_value *= (n_zw_geo[k][wid] + idx + eta)/(n_z_geo[k] + b + eta0)
    #             #         b += 1
    #             # bb*=rule_value
    #             # b = 1
    #             # rule_value = 1.0
    #             # for ent_w, w_count in ent_freq_dict.items():
    #             #     for idx in range(1, w_count + 1):
    #             #         wid = ent_word_id_dict[ent_w]['id']
    #             #         rule_value *= (n_zw_ent[k][wid] + idx + beta)/(n_z_ent[k] + b + beta0)
    #             #         b += 1
    #             # bb *= rule_value
    #             # b = 1
    #             # rule_value = 1.0
    #             # for key_w, w_count in key_freq_dict.items():
    #             #     for idx in range(1, w_count + 1):
    #             #         wid = key_word_id_dict[key_w]['id']
    #             #         rule_value *= (n_zw_key[k][wid] + idx + lambd)/(n_z_key[k] + b + lambd0)
    #             #         b += 1
    #             # bb *= rule_value
    #             # print(aa-bb)
    #
    #             # if rule_value < smallDouble:
    #             #     underflowcount[k] -= 1
    #             #     rule_value *= largeDouble
    #             # prob = recompute(prob, underflowcount)
    #         return au.sample_index_by_array_value(np.array(prob))
    #
    #     def update_using_freq_dict(tw_freq_dict_, word_id_dict, n_z_, n_zw_, factor):
    #         for w, w_freq in tw_freq_dict_.items():
    #             w_id = word_id_dict[w]['id']
    #             w_freq *= factor
    #             n_z_[cluster] += w_freq
    #             n_zw_[cluster][w_id] += w_freq
    #
    #     for i in range(iter_num):
    #         if iter_num > 10:
    #             print(str(i) + '\t' + str(m_z) + '\n' if i % int((iter_num + 1) / 10) == 0 else '', end='')
    #         for d in range(D):
    #             tw = twarr[d]
    #             tw_geo_freq_dict = tw['geo']
    #             tw_ent_freq_dict = tw['ent']
    #             tw_key_freq_dict = tw['key']
    #
    #             cluster = z[d]
    #             m_z[cluster] -= 1
    #             # for geo_w, geo_w_freq in tw_geo_freq_dict.items():
    #             #     geo_w_id = geo_words[geo_w]['id']
    #             #     n_z_geo[cluster] -= geo_w_freq
    #             #     n_zw_geo[cluster][geo_w_id] -= geo_w_freq
    #             # for ent_w, ent_w_freq in tw_ent_freq_dict.items():
    #             #     ent_w_id = ent_words[ent_w]['id']
    #             #     n_z_ent[cluster] -= ent_w_freq
    #             #     n_zw_ent[cluster][ent_w_id] -= ent_w_freq
    #             # for key_w, key_w_freq in tw_key_freq_dict.items():
    #             #     key_w_id = key_words[key_w]['id']
    #             #     n_z_key[cluster] -= key_w_freq
    #             #     n_zw_key[cluster][key_w_id] -= key_w_freq
    #             update_using_freq_dict(tw_geo_freq_dict, geo_word_id_dict, n_z_geo, n_zw_geo, -1)
    #             update_using_freq_dict(tw_ent_freq_dict, ent_word_id_dict, n_z_ent, n_zw_ent, -1)
    #             update_using_freq_dict(tw_key_freq_dict, key_word_id_dict, n_z_key, n_zw_key, -1)
    #
    #             cluster = sample_cluster(tw)
    #
    #             z[d] = cluster
    #             m_z[cluster] += 1
    #             update_using_freq_dict(tw_geo_freq_dict, geo_word_id_dict, n_z_geo, n_zw_geo, 1)
    #             update_using_freq_dict(tw_ent_freq_dict, ent_word_id_dict, n_z_ent, n_zw_ent, 1)
    #             update_using_freq_dict(tw_key_freq_dict, key_word_id_dict, n_z_key, n_zw_key, 1)
    #
    #     tw_topic_arr = [[] for _ in range(K)]
    #     for d in range(D):
    #         tw_topic_arr[z[d]].append(twarr[d])
    #
    #     def print_top_freq_word_in_dict(word_id_dict, n_zw_, cluster):
    #         c = 1
    #         freq_list = [[word, n_zw_[cluster][word_id_dict[word]['id']]]
    #                      for word in word_id_dict.keys() if
    #                      n_zw_[cluster][word_id_dict[word]['id']] >= 10]
    #         for pair in sorted(freq_list, key=lambda x: x[1], reverse=True)[:20]:
    #             print('{:<15}{:<5}'.format(pair[0], pair[1]), end='\n' if c % 5 == 0 else '\t')
    #             c += 1
    #         print('\n' if (c - 1) % 5 != 0 else '', end='')
    #
    #     for i, twarr in enumerate(tw_topic_arr):
    #         if len(twarr) == 0:
    #             continue
    #         print('\ncluster', i, '\t\ttweet number', len(twarr))
    #         print('-geo')
    #         print_top_freq_word_in_dict(geo_word_id_dict, n_zw_geo, i)
    #         print('-ent')
    #         print_top_freq_word_in_dict(ent_word_id_dict, n_zw_ent, i)
    #         print('-key')
    #         print_top_freq_word_in_dict(key_word_id_dict, n_zw_key, i)
    #
    #     return tw_topic_arr, z


class TempObject:
    def __init__(self, freqcounter):
        self.freqcounter = freqcounter

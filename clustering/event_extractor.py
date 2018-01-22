from collections import Counter

import numpy as np
import pandas as pd

import utils.function_utils as fu
from seeding.event_classifier import LREventClassifier
from clustering.cluster_service import ClusterService
from utils.id_freq_dict import IdFreqDict
from clustering.gsdmm_semantic_stream import SemanticStreamClusterer
from clustering.gsdpmm_stream import GSDPMMStreamClusterer
from clustering.gsdpmm_semantic_stream import GSDPMMSemanticStreamClusterer
# from clustering.gsdpmm_semantic_stream_static import GSDPMMSemanticStreamStatic


class EventExtractor:
    def __init__(self, dict_file, model_file):
        self.freqcounter = self.classifier = self.filteredtw = None
        # self.construct(dict_file, model_file)
    
    def construct(self, dict_file, model_file):
        self.freqcounter = IdFreqDict()
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
    
    def show_weight_of_words(self):
        thetaE = self.classifier.get_theta()[0]
        table = pd.DataFrame.from_dict(self.freqcounter.worddict.dictionary).T
        table.drop(axis=1, labels=['df'], inplace=True)
        table.sort_values(by='idf', ascending=False, inplace=True)
        for index, series in table.iterrows():
            table.loc[index, 'theta'] = thetaE[0][int(series['id'])]
        print(table)
    
    # def TOT_twarr_with_label(self, twarr, tw_cluster_label):
    #     a,b,c,d = self.TOT_twarr(twarr, 0.01, 0.05, 30, 20, tw_cluster_label)
    #     # print(self.cluster_label_prediction_table(tw_cluster_label, b))
    #     e,f,g,h = self.event_table_recall(tw_cluster_label, b)
    #     print(e)
    #     print(f)
    #     return
    #     base_path = getconfig().dc_test + 'TOT/'
    #     alpha_range = beta_range = gamma_range = [i / 100 for i in range(1, 10, 3)] + \
    #                                              [i / 10 for i in range(1, 10, 3)]
    #     K_range = [20, 30, 40, 50]
    #     """cluster using different hyperparams in multiprocess way"""
    #     iter_num = 3
    #     process_num = 19
    #     hyperparams = [(a, b, k) for a in alpha_range for b in beta_range for k in K_range]
    #     param_num = len(hyperparams)
    #     tempobj = TempObject(self.freqcounter)
    #     params = [(tempobj, twarr, *param, iter_num, tw_cluster_label) for param in hyperparams]
    #     res_list = self.clustering_multi(EventExtractor.TOT_twarr, params, process_num)
    #     frame = pd.DataFrame(index=np.arange(0, param_num), columns=['alpha', 'beta', 'K'])
    #     for i in range(param_num):
    #         frame.loc[i] = hyperparams[i]
    #     """start plotting figures"""
    #     for (alpha, K), indices in frame.groupby(['alpha', 'K']).groups.items():
    #         fig = plt.figure()
    #         fig.set_figheight(8)
    #         fig.set_figwidth(8)
    #         for i in indices:
    #             beta = frame.loc[i]['beta']
    #             gamma = frame.loc[i]['gamma']
    #             key_distrb, ht_distrb, tw_cluster_pred, iter_x, nmi_y = res_list[i]
    #             plt.plot(iter_x, nmi_y, '-', label='beta=' + str(beta) + ',gamma=' + str(gamma))
    #         title = 'alpha=' + str(alpha) + ',K=' + str(K)
    #         plt.title(title)
    #         plt.ylabel('NMI')
    #         plt.ylim(0.25, 0.75)
    #         plt.legend(loc='lower left')
    #         plt.text(iter_num * 0.6, 0.70,
    #                  'final nmi: ' + str(round(max([res_list[i][4][-1] for i in indices]), 4)), fontsize=15)
    #         plt.savefig('TOT_' + title + '.png')
    #
    #     top_ramk = 20
    #     alpha_idx = 0
    #     beta_idx = 1
    #     # gamma_idx = 2
    #     K_idx = 2
    #     tw_cluster_pred_idx = 5
    #     nmi_idx = 7
    #     table_idx = 8
    #     recall_idx = 9
    #
    #     event_cluster_label = [i for i in range(pos_cluster_num)]
    #     summary_list = [hyperparams[i] + res_list[i] +
    #                     self.event_table_recall(tw_cluster_label, res_list[i][2], event_cluster_label)
    #                     for i in range(param_num)]
    #     top_recall_summary_list = [summary_list[i] for i in
    #                                np.argsort([summary[recall_idx] for summary in summary_list])[:top_ramk:-1]]
    #     top_nmi_summary_list = [summary_list[i] for i in
    #                             np.argsort([summary[nmi_idx][-1] for summary in summary_list])[:top_ramk:-1]]
    #
    #     def dump_cluster_info(summary_list_, path):
    #         for rank, summary in enumerate(summary_list_):
    #             res_dir = path + '{}_recall_{:0<6}_nmi_{:0<6}_alpha_{:0<6}_beta_{:0<6}_K_{}/'. \
    #                 format(rank, round(summary[recall_idx], 6), round(summary[nmi_idx][-1], 6),
    #                        summary[alpha_idx], summary[beta_idx], summary[K_idx])
    #             fu.makedirs(res_dir)
    #             tw_topic_arr = self.create_clusters_with_labels(twarr, summary[tw_cluster_pred_idx])
    #             for i, _twarr in enumerate(tw_topic_arr):
    #                 if not len(_twarr) == 0:
    #                     fu.dump_array(res_dir + str(i) + '.txt',
    #                                   [tw[TweetKeys.key_cleantext] for tw in _twarr])
    #             table = summary[table_idx]
    #             table.to_csv(res_dir + 'table.csv')
    #
    #     top_recall_path = base_path + 'max_recalls/'
    #     fu.rmtree(top_recall_path)
    #     dump_cluster_info(top_recall_summary_list, top_recall_path)
    #     top_nmi_path = base_path + 'max_nmis/'
    #     fu.rmtree(top_nmi_path)
    #     dump_cluster_info(top_nmi_summary_list, top_nmi_path)
    #     return 0, 0
    #
    # def TOT_twarr(self, twarr, alpha, beta, K, iter_num, ref_labels=None):
    #     key_ner_pos_token = TweetKeys.key_wordlabels
    #     key_normed_timestamp = TweetKeys.key_normed_timestamp
    #     key_tw_word_freq = 'dup'
    #
    #     all_timestamp = [du.get_timestamp_form_created_at(tw[TweetKeys.key_created_at]) for tw in twarr]
    #     mintstamp = int(min(all_timestamp) - 1e4)
    #     maxtstamp = int(max(all_timestamp) + 1e4)
    #     twarr = twarr[:]
    #     words = dict()
    #     rewords = dict()
    #     """pre-process the tweet text, include dropping non-common terms"""
    #     for tw in twarr:
    #         tw_timestamp = du.get_timestamp_form_created_at(tw[TweetKeys.key_created_at])
    #         tw[key_normed_timestamp] = (tw_timestamp - mintstamp) / (maxtstamp - mintstamp)
    #         wordlabels = tw[key_ner_pos_token]
    #         for i in range(len(wordlabels) - 1, -1, -1):
    #             wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
    #             if not self.freqcounter.is_valid_keyword(wordlabels[i][0]):
    #                 del wordlabels[i]
    #             else:
    #                 word = wordlabels[i][0]
    #                 if word in words:
    #                     words[word]['freq'] += 1
    #                 else:
    #                     words[word] = {'freq': 1, 'id': len(words.keys())}
    #     min_df = 4
    #     for w in list(words.keys()):
    #         if words[w]['freq'] < min_df:
    #             del words[w]
    #     for idx, w in enumerate(sorted(words.keys())):
    #         words[w]['id'] = idx
    #         rewords[idx] = w
    #     for tw in twarr:
    #         tw[key_tw_word_freq] = dict(Counter([wlb[0] for wlb in tw[key_ner_pos_token] if wlb[0] in words]))
    #     # Validation
    #     for i in range(max(ref_labels) + 1):
    #         n_tstamp_arr = [twarr[d][key_normed_timestamp] for d in range(len(twarr)) if ref_labels[d] == i]
    #         print('{:<3}'.format(i), np.mean(n_tstamp_arr))
    #     """definitions of parameters"""
    #     D = len(twarr)
    #     V = words.__len__()
    #     alpha0 = K * alpha
    #     beta0 = V * beta
    #     z = [0] * D
    #     m_z = [0] * K
    #     n_z = [0] * K
    #     n_zw = [[0] * V for _ in range(K)]
    #     psi_z = [[0, 0] for _ in range(K)]
    #     """initialize the counting arrays"""
    #     for d in range(D):
    #         cluster = int(K * np.random.random())
    #         z[d] = cluster
    #         m_z[cluster] += 1
    #         for word, freq in twarr[d][key_tw_word_freq].items():
    #             n_z[cluster] += freq
    #             n_zw[cluster][words[word]['id']] += freq
    #     """recalculate psi_1 & psi_2 of each cluster"""
    #     def recalculate_psi(cluster):
    #         normed_timestamps_arr = [twarr[d][key_normed_timestamp] for d in range(len(z)) if z[d] == cluster]
    #         sample_num = len(normed_timestamps_arr)
    #         if sample_num == 0:
    #             psi_z[cluster][0] = psi_z[cluster][1] = -1
    #         elif sample_num == 1:
    #             psi_z[cluster][0] = normed_timestamps_arr[0]
    #             psi_z[cluster][1] = 1 - normed_timestamps_arr[0]
    #         else:
    #             mean = np.mean(normed_timestamps_arr)
    #             var_sq = np.var(normed_timestamps_arr)
    #             base = mean * (1 - mean) / var_sq - 1
    #             base = 1
    #             # print(mean, var_sq, base)
    #             psi_z[cluster][0] = mean * base
    #             psi_z[cluster][1] = (1 - mean) * base
    #     """make sampling using current counting information"""
    #     def sample_cluster(tw, cur_iter=None):
    #         tstamp = tw[key_normed_timestamp]
    #         prob = [0] * K
    #         for k in range(K):
    #             prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
    #             psi1_k, psi2_k = psi_z[k]
    #             if not (psi1_k == -1 or psi2_k == -1):
    #                 prob[k] *= ((1 - tstamp) ** (psi1_k - 1)) * (tstamp ** (psi2_k - 1)) / \
    #                            fu.B_function(psi1_k, psi2_k)
    #             rule_value = 1.0
    #             i = 0
    #             freq_dict = tw[key_tw_word_freq]
    #             for w, freq in freq_dict.items():
    #                 for j in range(freq):
    #                     rule_value *= (n_zw[k][words[w]['id']] + beta + j) / (n_z[k] + beta0 + i)
    #                     i += 1
    #             prob[k] *= rule_value
    #         return np.argmax(prob) if (cur_iter is not None and cur_iter > iter_num - 5) \
    #             else au.sample_index_by_array_value(np.array(prob))
    #     """start iteration"""
    #     for k in range(K):
    #         recalculate_psi(k)
    #     iter_x = list()
    #     nmi_y = list()
    #     for i in range(iter_num):
    #         print(i, m_z)
    #         for d in range(D):
    #             cluster = z[d]
    #             z[d] = -1
    #             m_z[cluster] -= 1
    #             recalculate_psi(cluster)
    #             freq_dict = twarr[d][key_tw_word_freq]
    #             for word, freq in freq_dict.items():
    #                 n_z[cluster] -= freq
    #                 n_zw[cluster][words[word]['id']] -= freq
    #
    #             new_cluster = sample_cluster(twarr[d], i)
    #
    #             z[d] = new_cluster
    #             m_z[new_cluster] += 1
    #             recalculate_psi(new_cluster)
    #             for word, freq in freq_dict.items():
    #                 n_z[new_cluster] += freq
    #                 n_zw[new_cluster][words[word]['id']] += freq
    #
    #         if ref_labels is not None:
    #             iter_x.append(i)
    #             nmi_y.append(au.score(z, ref_labels, score_type='nmi'))
    #
    #     if ref_labels is not None:
    #         return n_zw, z, iter_x, nmi_y
    #     else:
    #         return n_zw, z
    
    # @staticmethod
    # def batch_cluster_with_label(twarr, label):
    #     from clustering.gsdmm import GSDMM
    #     GSDMM.input_twarr_with_label(twarr, label)
    #     # print('batch num', len(tw_batches), 'average batch size', np.mean([len(tw) for tw in tw_batches]))
    #     # K = 40
    #     # hold_batch_num = 5
    #     # s = SemanticStreamClusterer(hold_batch_num=hold_batch_num)
    #     # s.set_hyperparams(alpha=0.05, etap=0.1, etac=0.1, etav=0.05, etah=0.1, K=K)
    #     # z_batch, lb_batch = list(), list()
    #     # """stream & online clustering"""
    #     # for idx, tw_batch in enumerate(tw_batches):
    #     #     cur_z, new_z, cur_lb = s.input_batch_with_label(tw_batches[idx], lb_batches[idx])
    #     #     if cur_z and new_z:
    #     #         z_batch.append(cur_z)
    #     #         lb_batch.append(cur_lb)
    #     #     print('\r' + ' ' * 20 + '\r', idx + 1, '/', len(tw_batches), 'groups, with totally',
    #     #           sum([len(t) for t in tw_batches[:idx + 1]]), 'tws processed', end='', flush=True)
    #     # print()
    #     # """evaluate the procedure"""
    #     # lb = fu.merge_list(lb_batches)
    #     # non_event_ids = [max(lb)]
    #     # evolution = pd.DataFrame(columns=range(K))
    #     # for batch_id in range(len(z_batch)):
    #     #     df = ClusterService.cluster_label_prediction_table(lb_batch[batch_id], z_batch[batch_id],
    #     #                                                        lbl_range=range(max(lb) + 1), pred_range=range(K))
    #     #     evolution.loc[batch_id, range(K)] = [df.columns[np.argmax(df.values[i])] if max(df.values[i]) > 0
    #     #         and df.columns[np.argmax(df.values[i])] not in non_event_ids else '' for i in range(len(df.values))]
    #     #     evolution.loc[batch_id, ' '] = ' '
    #     #     for event_id, event_cnt in dict(Counter(lb_batch[batch_id])).items():
    #     #         if event_id not in non_event_ids:
    #     #             evolution.loc[batch_id, 'e' + str(event_id)] = event_cnt
    #     # for batch_id in range(len(z_batch)):
    #     #     for event_id, event_cnt in dict(Counter(lb_batch[batch_id])).items():
    #     #         if event_id in non_event_ids:
    #     #             evolution.loc[batch_id, 'e' + str(event_id)] = event_cnt
    #     #
    #     # evolution_ = evolution.loc[:, range(K)]
    #     # detected = sorted(set(fu.merge_list(evolution_.values.tolist())).difference({''}))
    #     # detected = [int(d) for d in detected]
    #     # num_detected = len(detected)
    #     # totalevent = sorted(set(lb).difference(set(non_event_ids)))
    #     # num_total = len(totalevent)
    #     #
    #     # evolution.loc[' '] = ' '
    #     # evolution.loc['info', evolution.columns[0]] = 'K={}, batch_size={}, batch_num={}, hold_batch={}, ' \
    #     #                                               'total tweet num={}'.\
    #     #     format(K, round(np.mean([len(tw) for tw in tw_batches]), 2), len(tw_batches),
    #     #            hold_batch_num, len(fu.merge_list(tw_batches)))
    #     # evolution.loc['detected', evolution.columns[0]] = detected
    #     # evolution.loc['totalevent', evolution.columns[0]] = totalevent
    #     # evolution.loc['recall', evolution.columns[0]] = str(num_detected) + '/' + str(num_total) + \
    #     #                                                 '=' + str(num_detected / num_total)
    #     # evolution.to_csv('table.csv')
    
    # @staticmethod
    # def analyze_batch():
    #     z_iter = fu.load_array('z_static_iter.txt')
    #     l_iter = fu.load_array('lb_static_iter.txt')
    
    @staticmethod
    def stream_cluster_with_label(tw_batches, lb_batches):
        # g = GSDPMMSemanticStreamStatic(hold_batch_num=4)
        # g = GSDPMMSemanticStreamClusterer(hold_batch_num=4)
        # g.set_hyperparams(900, 0.1, 0.1, 0.1, 0.1)
        
        g = GSDPMMStreamClusterer(hold_batch_num=4)
        g.set_hyperparams(100, 0.05)
        
        # g = SemanticStreamClusterer(hold_batch_num=4)
        # g.set_hyperparams(1, 0.05, 0.05, 0.05, 0.05, 80)
        
        z_evo, l_evo, s_evo = list(), list(), list()
        for pred_cluid in range(len(tw_batches)):
            print('\r{}\r{}/{} groups, {} tws processed'.format(' ' * 30, pred_cluid + 1, len(tw_batches),
                  sum([len(t) for t in tw_batches[:pred_cluid + 1]])), end='', flush=True)
            z, label = g.input_batch_with_label(tw_batches[pred_cluid], lb_batches[pred_cluid])
            if not z:
                continue
            if not len(label) == len(z):
                raise ValueError('length inconsistent')
            z_evo.append([int(i) for i in z])
            l_evo.append([int(i) for i in label])
            s = g.clusters_similarity()
            s_evo.append(s)
            # print('differ:', set(s.keys()).difference(set(z)))
        fu.dump_array('evolution.txt', [z_evo, l_evo, s_evo])
        print(g.get_hyperparams_info())
        return z_evo, l_evo, s_evo
    
    @staticmethod
    def analyze_stream(z_evo=None, l_evo=None, s_evo=None, rep_score=0.7):
        """ evaluate the clustering results """
        if not z_evo or not l_evo or not s_evo:
            z_evo, l_evo, s_evo = fu.load_array('evolution.txt')
        
        history_z = set([z for z in fu.merge_list(z_evo)])
        print('\nhistory_z {}, effective cluster number {}'.format(sorted(history_z), len(history_z)))
        
        all_labels = fu.merge_list(l_evo)
        ne_cluid = int(max(all_labels))
        evo = pd.DataFrame(dtype=int)
        for batch_id in range(len(z_evo)):
            # s_batch is the similarity of every predicted cluster in an iteration
            z_batch, lb_batch, s_batch = z_evo[batch_id], l_evo[batch_id], s_evo[batch_id]
            for k in list(s_batch.keys()):
                s_batch[int(k)] = s_batch[k]
                s_batch.pop(k)
            df = ClusterService.cluster_prediction_table([int(i) for i in lb_batch], [int(i) for i in z_batch])
            for pred_cluid, row in df.iterrows():
                row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
                rep_cluid = int(row.index[row_max_idx])
                rep_twnum = int(row[rep_cluid])
                similarity = round(s_batch[pred_cluid], 3)
                if row.loc[rep_cluid] == 0 or row.loc[rep_cluid] < row_sum * rep_score or rep_cluid == ne_cluid:
                    evo.loc[batch_id, pred_cluid] = '{: <4},{:0<5},{}'.format(ne_cluid, similarity, rep_twnum)
                else:
                    evo.loc[batch_id, pred_cluid] = '{: <4},{:0<3},{}'.format(rep_cluid, similarity, rep_twnum)
        
        evo.fillna('', inplace=True)
        real_event_ids = fu.merge_list(l_evo)
        real_event_num = Counter(real_event_ids)
        detected_event_ids = [int(item.split(',')[0]) for item in fu.merge_list(evo.values.tolist())
                              if item != '' and int(item.split(',')[0]) != ne_cluid]
        detected_event_num = Counter(detected_event_ids)
        detected_event_id = [d for d in sorted(set(detected_event_ids))]
        num_detected = len(detected_event_id)
        event_id_corpus = sorted(set(all_labels).difference({ne_cluid}))
        num_corpus = len(event_id_corpus)
        
        first_col = evo.columns[0]
        evo.loc['eventdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, detected_event_num[eventid])
                                                             for eventid in sorted(detected_event_num.keys())])
        evo.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, real_event_num[eventid])
                                                            for eventid in sorted(real_event_num.keys())])
        evo.loc['detectedid', first_col] = str(detected_event_id)
        evo.loc['totalevent', first_col] = str(event_id_corpus)
        evo.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
        evo.fillna('', inplace=True)
        evo.to_csv('table.csv')
        print(detected_event_id, num_detected / num_corpus)
    
    @staticmethod
    def grid_stream_cluster_with_label_multi(tw_batches, lb_batches):
        print('batch num:{}, batch size:{}'.format(len(tw_batches), len(tw_batches[0])))
        
        a_range = [i for i in range(1, 10, 5)] + [i*10 for i in range(1, 10, 3)] + [i*100 for i in range(1, 10, 2)]
        b_range = [i/1000 for i in range(1, 10, 2)] + [i/100 for i in range(1, 10, 2)]
        params = [(a, b) for a in a_range for b in b_range]
        res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
                       [(tw_batches, lb_batches, GSDPMMStreamClusterer(4), p) for p in params])
        
        # a_range = [1, 0.1]
        # k_range = [30, 50]
        # p_range = c_range = v_range = h_range = [0.01, 0.1, 1]
        # params = [(a, p, c, v, h, k) for a in a_range for p in p_range
        #           for c in c_range for v in v_range for h in h_range for k in k_range]
        # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
        #                 [(tw_batches, lb_batches, SemanticStreamClusterer(4), *p) for p in params])
        
        for idx, res in enumerate(sorted(res_list, key=lambda x: x[-1], reverse=True)):
            print(res[2:5])
    
    @staticmethod
    def stream_cluster_with_label_multi(tw_batches, lb_batches, clusterer, params):
        clusterer.set_hyperparams(*params)
        
        z_evo, l_evo = list(), list()
        for pred_cluid in range(len(tw_batches)):
            z, label = clusterer.input_batch_with_label(tw_batches[pred_cluid], lb_batches[pred_cluid])
            if not z:
                continue
            if not len(label) == len(z):
                raise ValueError('length inconsistent')
            z_evo.append([int(i) for i in z])
            l_evo.append([int(i) for i in label])
        
        """ evaluation """
        all_labels = fu.merge_list(l_evo)
        rep_score = 0.7
        non_event_label = int(max(all_labels))
        evolution = pd.DataFrame(dtype=int)
        for batch_id in range(len(z_evo)):
            z_batch, lb_batch = z_evo[batch_id], l_evo[batch_id]
            df = ClusterService.cluster_prediction_table([int(item) for item in lb_batch],
                                                         [int(item) for item in z_batch])
            for pred_cluid, row in df.iterrows():
                row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
                max_cluid = int(row.index[row_max_idx])
                if row[max_cluid] == 0 or row[max_cluid] < row_sum * rep_score or max_cluid == non_event_label:
                    evolution.loc[batch_id, pred_cluid] = non_event_label
                else:
                    evolution.loc[batch_id, pred_cluid] = max_cluid
        
        evolution.fillna(non_event_label, inplace=True)
        real_event_num = Counter(all_labels)
        detected_event_ids = [int(item) for item in fu.merge_list(evolution.values.tolist())
                              if item != non_event_label]
        detected_event_num = Counter(detected_event_ids)
        detected_event_id = [d for d in sorted(set(detected_event_ids))]
        num_detected = len(detected_event_id)
        event_id_corpus = sorted(set(all_labels).difference({non_event_label}))
        num_corpus = len(event_id_corpus)
        
        # first_col = evolution.columns[0]
        # evolution.loc['eventdistrb', first_col] = '  '.join(
        #     ['{}:{}'.format(eventid, detected_event_num[eventid])
        #      for eventid in sorted(detected_event_num.keys())])
        # evolution.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, real_event_num[eventid])
        #                                                     for eventid in sorted(real_event_num.keys())])
        # evolution.loc['detectedid', first_col] = str(detected_event_id)
        # evolution.loc['totalevent', first_col] = str(event_id_corpus)
        # evolution.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
        # evolution.fillna('', inplace=True)
        
        info = clusterer.get_hyperparams_info()
        return z_evo, l_evo, info, num_detected / num_corpus

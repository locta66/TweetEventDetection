import math
from collections import Counter

from config.configure import getcfg
import utils.tweet_keys as tk
import utils.array_utils as au
import utils.function_utils as fu
from clustering.cluster_service import ClusterService

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GSDMM:
    @staticmethod
    def input_twarr_with_label(twarr, label):
        alpha_range = beta_range = [i/100 for i in range(1, 10, 3)] + [i/10 for i in range(1, 10, 3)] + \
                                   [i for i in range(1, 10, 3)]
        K_range = [20, 30, 40, 50]
        """cluster using different hyperparams in multiprocess way"""
        iter_num = 100
        process_num = 20
        hyperparams = [(a, b, K) for a in alpha_range for b in beta_range for K in K_range]
        res_list = list()
        for i in range(int(math.ceil(len(hyperparams) / process_num))):
            param_list = [(None, twarr, *param, iter_num, label) for param in
                          hyperparams[i * process_num: (i + 1) * process_num]]
            res_list += fu.multi_process(GSDMM.GSDMM_twarr, param_list)
            print('{:<4} /'.format((i + 1) * process_num), len(hyperparams), 'params processed')
        """group the data by alpha and K"""
        frame = pd.DataFrame(index=np.arange(0, len(hyperparams)), columns=['alpha', 'beta', 'K'])
        for i in range(len(hyperparams)):
            frame.loc[i] = hyperparams[i]
        print('\n', frame, '\n')
        """start plotting figures"""
        for (alpha, K), indices in frame.groupby(['alpha', 'K']).groups.items():
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(8)
            for i in indices:
                beta = frame.loc[i]['beta']
                topic_word_dstrb, tw_cluster_pred, iter_x, nmi_y, homo_y, cmplt_y = res_list[i]
                # print(alpha, beta, K, ':', np.array(indices))
                plt.plot(iter_x, nmi_y, '-', lw=1.5, label='beta=' + str(round(beta, 2)))
            plt.xlabel('iteration')
            plt.ylabel('NMI')
            plt.ylim(0.25, 0.75)
            plt.title('alpha=' + str(round(alpha, 2)) + ',K=' + str(K))
            plt.legend(loc='lower right')
            plt.grid(True, '-', color='#333333', lw=0.8)
            plt.text(iter_num - 20, 0.70, 'final nmi: ' + str(round(max([res_list[i][3][-1] for i in indices]), 6)),
                     fontsize=15, verticalalignment='bottom', horizontalalignment='left')
            plt.savefig(getcfg().dc_test + 'alpha=' + str(round(alpha, 2)) + '_K=' + str(K) + '.png')
        return None, None
    
    @staticmethod
    def GSDMM_twarr(twarr, alpha, beta, K, iter_num):
        ner_pos_token = tk.key_wordlabels
        twarr = twarr[:]
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                if not ClusterService.is_valid_keyword(wordlabels[i][0]):
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
        z_iter = list()
        for i in range(iter_num):
            z_iter.append(z[:])
            
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
        
        z_iter.append(z[:])
        return z_iter

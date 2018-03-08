# from ChatFilter import ChatFilter
#     c = ChatFilter()
#     c.set_twarr(twarr)
#     c.set_hyperparams(0.9, 0.01, 60)
#     n_zw, z = c.recluster_using_GSDMM()
#     import pandas as pd
#     cluster_table = pd.DataFrame(index=set(z), columns=set(label), data=0)
#     for i in range(len(label)):
#         cluster_table.loc[z[i]][label[i]] += 1
#     print(cluster_table)
#     for i in range(10):
#         print(z[2], c.sample_cluster(twarr[2]))
#     for i in range(10):
#         print(z[100], c.sample_cluster(twarr[100]))
#     for i in range(10):
#         print(z[2314], c.sample_cluster(twarr[2314]))

# from ChatFilter import ChatFilter
# c = ChatFilter()
# c.set_twarr(twarr)                            # twarr = [{tweet1}, {tweet2}, ...]
# c.set_hyperparams(0.9, 0.01, 60)              # 推荐超参，论文里用的是alpha=0.1 * len(twarr), beta=0.02
# 各个聚类的词分布, 各条推文的预测聚类 = c.recluster_using_GSDMM()
# 一条推文的预测聚类 = c.sample_cluster(tw)      # tw = {tweet}, 第二个参数不使用

# 如果需要转化成表格，可以pandas
# label = 各条推文的实际分类
# table = pd.DataFrame(index=set(各条推文的预测聚类), columns=set(label), data=0)
# for i in range(len(label)):
#     table.loc[每条推文的预测聚类[i]][label[i]] += 1
# print(cluster_table)


# from sklearn.metrics import normalized_mutual_info_score
# metrics.normalized_mutual_info_score(label, 各条推文的预测聚类)    # 可以用来看聚类相对参考标签的效果
import re
from collections import Counter

import numpy as np
from nltk.corpus import stopwords

# 处理的字段为'text'
key_text = 'text'


class ChatFilter:
    def __init__(self):
        self.twarr = self.words = None
        self.stopworddict = {}
        for stopword in stopwords.words('english'):
            self.stopworddict[stopword] = True
    
    def is_valid_keyword(self, word):
        if not word:
            return False
        startswithchar = re.search('^[^a-zA-Z#]', word) is None
        notsinglechar = re.search('^\w$', word) is None
        notstopword = word not in self.stopworddict
        return startswithchar and notsinglechar and notstopword
    
    def set_twarr(self, twarr):
        self.twarr = twarr[:]
        self.preprocess()
    
    def set_hyperparams(self, alpha, beta, iter_num):
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num
    
    def clean_text(self, text):
        word_tokens = text.split(' ')
        for i in range(len(word_tokens) - 1, -1, -1):
            # 或者其他词净化操作
            word_tokens[i] = word_tokens[i].lower().strip('#').strip()
            if not self.is_valid_keyword(word_tokens[i]):
                del word_tokens[i]
        return word_tokens
    
    def preprocess(self):
        twarr = self.twarr
        words = self.words = dict()
        key_tokens = 'tokens'
        for tw in twarr:
            word_tokens = self.clean_text(tw[key_text])
            for word in word_tokens:
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
            tw[key_tokens] = word_tokens
        min_df = 3
        for w in list(words.keys()):
            if words[w]['freq'] < min_df:
                del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([word for word in tw[key_tokens] if word in words]))
    
    def sample_cluster(self, tw, cur_iter=None):
        words = self.words
        alpha = self.alpha
        beta = self.beta
        iter_num = self.iter_num
        K = self.K
        D = self.D
        V = self.V
        beta0 = V * beta
        m_z = self.m_z
        n_z = self.n_z
        n_zw = self.n_zw
        prob = [0] * K
        
        if 'dup' in tw:
            freq_dict = tw['dup']
        else:
            freq_dict = dict(Counter([word for word in self.clean_text(tw[key_text]) if word in words]))
        
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
        array = prob + [new_cluster_prob]
        if cur_iter is not None and cur_iter > iter_num - 5:
            return np.argmax(array)
        else:
            return np.random.choice(a=[i for i in range(len(array))], p=np.array(array) / np.sum(array))
    
    def recluster_using_GSDMM(self):
        twarr = self.twarr
        words = self.words
        """definitions of parameters"""
        K = self.K = 1
        D = self.D = len(twarr)
        V = self.V = len(words.keys())
        iter_num = self.iter_num
        z = self.z = [0] * D
        m_z = self.m_z = [0] * K
        n_z = self.n_z = [0] * K
        n_zw = self.n_zw = [[0] * V for _ in range(K)]
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            for word, freq in twarr[d]['dup'].items():
                n_z[cluster] += freq
                n_zw[cluster][words[word]['id']] += freq
        """start iteration"""
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
                    self.K -= 1
                    for d_ in range(D):
                        if z[d_] > min_tw_num_idx:
                            z[d_] -= 1
            
                cluster = self.sample_cluster(twarr[d], i)
            
                if cluster >= self.K:
                    m_z.append(0)
                    n_z.append(0)
                    n_zw.append([0] * V)
                    self.K += 1
                
                z[d] = cluster
                m_z[cluster] += 1
                for word, freq in freq_dict.items():
                    n_z[cluster] += freq
                    n_zw[cluster][words[word]['id']] += freq
        
        return n_zw, z

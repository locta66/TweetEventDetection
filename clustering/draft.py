import multiprocessing as mp


pool = mp.Pool(processes=1)
res_getter = list()
for i in range(3):
    res = pool.apply_async(func=M.printmyself, args=(m, 'shit', dc))
    res_getter.append(res)

pool.close()
pool.join()
results = list()
for i in range(3):
    results.append(res_getter[i].get())

print(results)

import utils.file_iterator as fi
from utils.multiprocess_utils import multi_process
from utils.array_utils import merge_list
from sklearn.cluster import dbscan
import numpy as np
import Levenshtein

twarr = fi.load_array('/home/nfs/cdong/tw/testdata/pos_tweets.sum')
for tw in twarr:
    tw['temp'] = tw['text'].lower()

ndim = twarr.__len__()
mat = np.ones([ndim, ndim])
for i in range(ndim - 1):
    for j in range(i + 1, ndim):
        istr = twarr[i]['temp']
        jstr = twarr[j]['temp']
        dist = Levenshtein.distance(istr, jstr) + 1
        if max(len(istr), len(jstr)) / dist >= 5:
            mat[i, j] = mat[j, i] = 0

sum(sum(mat))
label, cluster = dbscan(mat, 0.5, 2, metric='precomputed')

for i in [2, 64, 23]:
    print(i, end=':')
    for idx, c in enumerate(cluster):
        if c == i:
            print(idx, end=' ')
    print()

ndim = twarr.__len__()
mmm = np.ones([ndim, ndim])


def twarr_dist_pairs_multi(twarr):
    for tw in twarr:
        tw['nouse'] = tw['text'].lower()
    total = len(twarr) - 1
    process_num = 16
    point_lists = [[i + 16 * j for j in range(int(total / process_num) + 1)
                    if (i + process_num * j) < total] for i in range(process_num)]
    pairs_blocks = multi_process(dist_pairs, [(twarr, point) for point in point_lists])
    for tw in twarr:
        del tw['nouse']
    return merge_list(pairs_blocks)


def dist_pairs(twarr, points):
    return merge_list([[(i, j, text_dist_less_than(twarr[i]['temp'], twarr[j]['temp']))
                        for j in range(i + 1, len(twarr))] for i in points])


def text_dist_less_than(text1, text2, threshold=0.2):
    dist = Levenshtein.distance(text1, text2) + 1
    if dist / max(len(text1), len(text2)) <= threshold:  # text 1 & 2 are similar
        return 0
    else:
        return 1


pairs = twarr_dist_pairs_multi(twarr)
for p in pairs:
    mmm[p[0]][p[1]] = mmm[p[1]][p[0]] = p[2]

sum(sum(mmm))
label, cluster = dbscan(mmm, 0.5, 2, metric='precomputed')

for i in [2, 64, 23]:
    print(i, end=':')
    for idx, c in enumerate(cluster):
        if c == i:
            print(idx, end=' ')
    print()

from sklearn.metrics import normalized_mutual_info_score

# y_pred = [0,0,1,1,2,2]
# y_true = [1,1,2,2,3,3]
y_pred = [1, 2, 1, 1, 1, ]
y_true = [5, 3, 5, 5, 5]

nmi = normalized_mutual_info_score(y_true, y_pred)
nmi

import pandas as pd

# pd.DataFrame(columns=['qwe', 'a', 'weqr','yt'], data=[[1,2,3,4], [0,0,0,0]])
data = np.array([[1, 2, 3],
                 [3, 4, 5],
                 [4, 5, 6],
                 [3, 2, 3],
                 [4, 3, 5],
                 ])
df = pd.DataFrame(data=data)
df.sort_values(by=[2, 0], ascending=False)
df.loc[:, 1]

for i, j in [1, 2, 3], [3, 4]:
    print(i, j)

# import re
# def preprocess(doc):
#     # pattern = re.compile(r'(\d\s\.\s\d)')
#     return re.sub(r'(\d\s\.\s\d)', '.', doc)
#
# for text in textarr[100:300]:
#     print(preprocess(text))


import sys
sys.path.append('../utils')
import utils.function_utils as fu
twarr = fu.load_array('/home/nfs/cdong/tw/seeding/NaturalDisaster/queried/NaturalDisaster.sum')
arr1=twarr[:2000]
arr2=twarr[2000:]
# cv = CV(analyzer='word', token_pattern=r'([a-zA-Z_-]+|\d+\.\d+|\d+)',
#         stop_words=stop_words, max_df=0.8, min_df=1e-5)


import re
print(re.findall(r'([a-zA-Z_-]+|\d+\.\d+|\d+)', ))

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


import re
# s = 'RT @bugwannostra: @Louuu_ thx		#FFFFs People power -_-      works	❤signing…		https://t.co/pl2bquE5Az'
s = 'RT @bugwannostra: @Louuu_432 thx 6.3 #FF-FFs, People power -_-  https:/ 2.34  234w.orks	❤signing… ht.tp:   https:'
re.findall(r'([a-zA-Z_-]+|\d+\.\d+|\d+)', s)



from sklearn.feature_extraction.text import CountVectorizer as CV
cv = CV(analyzer='word', token_pattern=r'([a-zA-Z_-]+|\d+\.\d+|\d+)',
        stop_words=stop_words, max_df=0.8, min_df=1e-5)
cv.fit_transform(textarr)


# 这一行的效果和直接运行cProfile.run("foo()")的显示效果是一样的
# p.strip_dirs().sort_stats(-1).print_stats()
# strip_dirs():从所有模块名中去掉无关的路径信息
# sort_stats():把打印信息按照标准的module/name/line字符串进行排序
# print_stats():打印出所有分析信息

# 按照在一个函数中累积的运行时间进行排序
# print_stats(3):只打印前3行函数的信息,参数还可为小数,表示前百分之几的函数信息

# python3.5 -m cProfile -o res event_extractor.py
import pstats
p = pstats.Stats("res")
p.strip_dirs( ).sort_stats("cumulative").print_stats(50)


内容 功能 效果 可视化


import spacy
nlp = spacy.load('en_core_web_lg')
doc = nlp('Large parts of this manual originate from Travis E. Oliphant’s book Guide to NumPy (which generously entered Public Domain in August 2008). The reference documentation for many of the functions are written by numerous contributors and developers of NumPy, both prior to and during the NumPy Documentation Marathon.')
[ent.label_ for ent in doc.ents]
print([(t.text, t.ent_type_, t.tag_) for t in doc])


import time

a = 1
s = time.time()
count = 0
for i in range(1 << 20):
    count += 1

print(time.time() -s)
s = time.time()

d = dict()
count = 0
for i in range(1 << 20):
    count += 1

print(time.time() -s)


class AA:
    def __init__(self):
        self.a = 0
    
    def __call__(self, *args, **kwargs):
        print('qwer')

def f(asd = AA()()):
    print('f')



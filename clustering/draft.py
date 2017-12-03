import multiprocessing as mp


class DC:
    def __init__(self):
        pass


class M:
    def __init__(self, value):
        self.value = value
    
    def printmyself(self, string, dc):
        return str(self.value) + string + str(type(dc))


m = M(233)
dc = DC()

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











import FileIterator as fi
from FunctionUtils import multi_process, merge_list
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
    if dist / max(len(text1), len(text2)) <= threshold:     # text 1 & 2 are similar
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
y_pred = [1,2,1,1,1,]
y_true = [5,3,5,5,5]


nmi = normalized_mutual_info_score(y_true, y_pred)
nmi


from collections import Counter

from extracting.extract_twarr_info import extract_twarr_full_info

from clustering.main2clusterer import get_tw_batches
from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import *
from clustering.event_extractor import hold_batch_num
from clustering.event_extractor import cluid_evo_file

import utils.array_utils as au
import utils.function_utils as fu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.tweet_utils as tu
import utils.timer_utils as tmu


def dump_cluinfo(file, cluid2cluster):
    except_attr = {tk.key_spacy}
    def drop_attribute(tw):
        new_tw = dict()
        for key in tw.keys():
            if key not in except_attr:
                new_tw[key] = tw[key]
        return new_tw
    
    cluid2info = dict()
    for cluid, cluster in cluid2cluster.items():
        if cluster.twnum <= 10:
            continue
        print('\n\n\n\n{}'.format(cluid))
        clu_twarr = cluster.get_twarr()
        full_info = extract_twarr_full_info(clu_twarr)
        full_info[-1] = tu.twarr_operate(full_info[-1], operation=drop_attribute)
        cluid2info[cluid] = full_info
    # fu.dump_array(file, [cluid2info])
    return cluid2info


def parse_cluinfo_readable(cluid2info):
    cluidarr = sorted(cluid2info.keys())
    for cluid in cluidarr:
        info = cluid2info[cluid]
        print(cluid)
        loc_freq_list, top_geo_info, hot, level, sorted_twarr = info
        
        print("\nloc freq list")
        print(loc_freq_list)
        print("\ntop geo information")
        for geo_info in top_geo_info:
            print(geo_info)
        print("\nhot: {}, level:{}".format(hot, level))
        print("\nsorted twarr")
        for idx, tw in enumerate(sorted_twarr):
            print(idx, tw[tk.key_text])
        print("\n\n\n")


def cluster_keyword_similarity_array(cludict, cluidarr):
    for cluid in cluidarr:
        cludict[cluid].extract_keywords()
    vec_label_arr = list()
    for i in range(0, len(cluidarr)):
        for j in range(0, len(cluidarr)):
            if i == j:
                continue
            cluster1, cluster2 = cludict[cluidarr[i]], cludict[cluidarr[j]]
            sim_vec = cluster1.word_sim_vec_with_other(cluster2)
            rep_lb1, rep_lb2 = cluster1.get_rep_label(0.8), cluster2.get_rep_label(0.8)
            label = rep_lb1 == rep_lb2 and rep_lb1 != -1
            vec_label_arr.append(np.concatenate([sim_vec, [label]], axis=0))
    return vec_label_arr


class Analyzer:
    def __init__(self, hold_batch_num):
        self.hold_batch_num = hold_batch_num
        self.tw_batches, self.cluid_batches = list(), list()
        self.cludict = dict()
    
    def set_batches(self, tw_batches, cluid_batches):
        self.tw_batches.clear()
        assert len(tw_batches) == len(cluid_batches)
        for batch_idx in range(len(tw_batches)):
            twarr, cluidarr = tw_batches[batch_idx], cluid_batches[batch_idx]
            assert len(twarr) == len(cluidarr)
            twharr = GSDPMMStreamIFDDynamic.pre_process_twarr(twarr)
            for idx in range(len(twharr)):
                twharr[idx].cluid = cluidarr[idx]
            self.tw_batches.append(twharr)
        self.cludict = dict([(cluid, ClusterHolder(cluid)) for cluid in set(au.merge_array(cluid_batches))])
    
    def start_iter(self):
        cludict = self.cludict
        hold_batch_num = self.hold_batch_num
        sep_pattern = '\n--\n\n'
        siminfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_siminfo.txt'
        textinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_textinfo.txt'
        scoreinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_scoreinfo.txt'
        batch_window = list()
        
        data = list()
        for batch_idx, twharr in enumerate(self.tw_batches):
            print('\r{}\rbatch idx {}'.format(' ' * 20, batch_idx), end='', flush=True)
            batch_window.append(twharr)
            for twh in twharr:
                twh.update_cluster(cludict[twh.cluid])
            if batch_idx < hold_batch_num:     # if tw number is not enough
                continue
            sim_info_arr = list()
            text_info_arr = list()
            score_info_arr = list()
            cluidarr = sorted([cluid for cluid, cluster in cludict.items() if cluster.twnum > 5])
            
            """ special operations """
            vec_label_arr = cluster_keyword_similarity_array(cludict, cluidarr)
            data.extend(vec_label_arr)
            
            oldest_twharr = batch_window.pop(0)
            for twh in oldest_twharr:
                twh.update_cluster(None)
            continue
            
            for cluid in cluidarr:
                cluster = cludict[cluid]
                clu_twarr = cluster.get_twarr()
                clu_lbarr = cluster.get_lbarr()
                clu_twnum = cluster.twnum
                cluster.rep_label = cluster.get_rep_label()
                
                cluster.extract_keywords()
                # clu_title_info = 'cluid: {}, label distrb: {}, rep_label: {}, is event score: {}\n'.\
                #     format(cluid, cluid2lbnum, str(clu_rep_label), cluster.get_event_score())
                #
                # for twh in clu_twarr:
                #     twh['detected'] = (twh[tk.key_event_label] == clu_rep_label)
                #
                # det_lb_txt = [(tw['detected'], tw[tk.key_event_label], tw[tk.key_text]) for tw in clu_twarr]
                # dlt_sort = sorted(det_lb_txt, key=lambda item: item[1])
                # text_info = '\n'.join(['  {:<2} {:<5} {}'.format(*item) for item in dlt_sort])
                #
                # score_info_arr.append(clu_title_info + sep_pattern)
                # text_info_arr.append(clu_title_info + text_info + sep_pattern)
            
            # fu.write_lines(textinfo_pattern.format(batch_idx), sim_info_arr)
            # fu.write_lines(textinfo_pattern.format(batch_idx), text_info_arr)
            # fu.write_lines(scoreinfo_pattern.format(batch_idx), score_info_arr)
            
            oldest_twharr = batch_window.pop(0)
            for twh in oldest_twharr:
                twh.update_cluster(None)
        
        data = np.array(data, dtype=np.float32)
        print(data.shape)
        np.save('./data_sim', data)


def train_keyword_similarity():
    from sklearn.linear_model import LogisticRegression
    from sklearn.externals import joblib
    np.random.seed(666)
    data = np.load('./data_sim.npy')
    data_len = len(data)
    sep_idx = int(data_len * 0.8)
    train, test = data[:sep_idx], data[sep_idx:]
    # rand_idx = au.shuffle([i for i in range(data_len)])
    # train, test = data[rand_idx[:sep_idx]], data[rand_idx[sep_idx:]]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    print(data.shape, train.shape, test.shape)
    
    if 1:
        c = LogisticRegression()
        c.fit(train_x, train_y)
        joblib.dump(c, 'temp_model')
    else:
        c = joblib.load('temp_model')
    
    print(c.coef_[0][:6].tolist(), '\n', c.coef_[0][6:12].tolist(), c.coef_[0][12])
    pred = c.predict_proba(test_x)[:, 1]
    print(au.score(test_y, pred, 'auc'))
    au.precision_recall_threshold(test_y, pred)


if __name__ == '__main__':
    # train_keyword_similarity()
    # exit()
    
    tmu.check_time('qwer')
    
    _tw_batches = get_tw_batches()
    _cluid_batches = fu.load_array(cluid_evo_file)
    
    tmu.check_time()
    _twarr = au.merge_array(_tw_batches)
    tu.twarr_nlp(_twarr)
    tmu.check_time(print_func=lambda dt: print('{} tweets spacy over in {} s'.format(len(_twarr), dt)))
    
    a = Analyzer(hold_batch_num)
    a.set_batches(_tw_batches, _cluid_batches)
    a.start_iter()
    
    train_keyword_similarity()
    
    tmu.check_time('qwer')

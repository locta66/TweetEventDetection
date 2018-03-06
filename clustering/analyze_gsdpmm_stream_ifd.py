from collections import Counter

from clustering.main2clusterer import get_tw_batches
from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import *
from clustering.event_extractor import hold_batch_num

import utils.timer_utils as tmu
import utils.function_utils as fu
import utils.tweet_keys as tk


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
        # cluinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_cluinfo.txt'
        sep_pattern = '\n--\n\n'
        siminfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_siminfo.txt'
        textinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_textinfo.txt'
        scoreinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_scoreinfo.txt'
        batch_window = list()
        for batch_idx, twharr in enumerate(self.tw_batches):
            print('\r{}batch idx {}\r'.format(' ' * 20, batch_idx), end='', flush=True)
            batch_window.append(twharr)
            for tw in twharr:
                cluster = self.cludict[tw.cluid]
                tw.update_cluster(cluster)
            if batch_idx < self.hold_batch_num:     # if tw number is not enough
                continue
            
            text_info_arr = list()
            score_info_arr = list()
            for cluid in sorted(self.cludict.keys()):
                cluster = self.cludict[cluid]
                if cluster.twnum == 0:
                    continue
                clu_twarr = cluster.get_twarr()
                clu_lbarr = cluster.get_lbarr()
                total_lbnum = len(clu_lbarr)
                cluid2lbnum = Counter(clu_lbarr)
                
                # label_info = 'label: {}\n'.format(cluid2lbnum)
                cluid_lbnum_s = sorted([(_cluid, _lbnum) for _cluid, _lbnum in cluid2lbnum.items()],
                                       key=lambda item: item[1], reverse=True)
                max_label, max_lbnum = cluid_lbnum_s[0][0], cluid_lbnum_s[0][1]
                rep_thres = 0.7
                if max_lbnum < total_lbnum * rep_thres:
                    clu_rep_label = None
                else:
                    clu_rep_label = max_label
                
                clu_title_info = 'cluid: {}, label distrb: {}, rep_label: {}, is event score: {}\n'.\
                    format(cluid, cluid2lbnum, str(clu_rep_label), cluster.get_event_score())
                
                for tw in clu_twarr:
                    tw['detected'] = True if tw[tk.key_event_label] == clu_rep_label else False
                
                det_lb_text_s = sorted([(tw['detected'], tw[tk.key_event_label], tw[tk.key_text])
                                        for tw in clu_twarr], key=lambda item: item[1])
                text_info = '\n'.join(['  {:>4} {:>5} {}'.format(*item) for item in det_lb_text_s])
                
                score_info_arr.append(clu_title_info + sep_pattern)
                text_info_arr.append(clu_title_info + text_info + sep_pattern)
            
            fu.write_lines(textinfo_pattern.format(batch_idx), text_info_arr)
            fu.write_lines(scoreinfo_pattern.format(batch_idx), score_info_arr)
            
            oldest_twharr = batch_window.pop(0)
            for tw in oldest_twharr:
                tw.update_cluster(None)


if __name__ == '__main__':
    """ load data """
    tw_batches = get_tw_batches()
    cluid_batches = fu.load_array('cluid_evo.txt')
    
    tmu.check_time()
    tu.twarr_nlp(au.merge_array(tw_batches))
    tmu.check_time(print_func=lambda dt: print('spacy over, time elapsed {} s'.format(dt)))
    
    a = Analyzer(hold_batch_num)
    a.set_batches(tw_batches, cluid_batches)
    a.start_iter()

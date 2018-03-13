from collections import Counter

from clustering.main2clusterer import get_tw_batches
from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import *
from clustering.event_extractor import hold_batch_num

import utils.array_utils as au
import utils.function_utils as fu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.timer_utils as tmu


# 若一个聚类只有一两条推文，不必急着将其去除，只需要将其保留
# 因为不能确定这个聚类什么时候就会成长，因此只需要在显示的时候将其屏蔽即可
# 若一直没有新的推文加入，这个推文很少的聚类自然会消失


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
    
    @staticmethod
    def importance_sort(twarr):
        docarr = [tw.get(tk.key_spacy) for tw in twarr]
        vecarr = np.array([np.concatenate(su.get_doc_pos_vectors(doc)) for doc in docarr])
        meanvec = np.mean(vecarr).reshape([1, -1])
        cos_with_mean = au.cosine_similarity(vecarr, meanvec).reshape(-1)
        sorted_idx = np.argsort(cos_with_mean)[::-1]
        return [twarr[idx] for idx in sorted_idx]
    
    def start_iter(self):
        cludict = self.cludict
        hold_batch_num = self.hold_batch_num
        # cluinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_cluinfo.txt'
        sep_pattern = '\n--\n\n'
        siminfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_siminfo.txt'
        textinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_textinfo.txt'
        scoreinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_scoreinfo.txt'
        batch_window = list()
        for batch_idx, twharr in enumerate(self.tw_batches):
            print('\r{}batch idx {}\r'.format(' ' * 20, batch_idx), end='', flush=True)
            batch_window.append(twharr)
            for twh in twharr:
                twh.update_cluster(cludict[twh.cluid])
            if batch_idx < hold_batch_num:     # if tw number is not enough
                continue
            
            text_info_arr = list()
            score_info_arr = list()
            cluidarr = sorted([cluid for cluid, cluster in cludict.items() if cluster.twnum > 2])
            for cluid in cluidarr:
                cluster = cludict[cluid]
                clu_twarr = cluster.get_twarr()
                clu_lbarr = cluster.get_lbarr()
                clu_twnum = cluster.twnum
                cluid2lbnum = Counter(clu_lbarr)
                
                # impt_sort_twarr = Analyzer.importance_sort(clu_twarr)
                
                rep_thres = 0.7
                max_label, max_lbnum = cluid2lbnum.most_common(1)[0]
                clu_rep_label = None if max_lbnum < clu_twnum * rep_thres else max_label
                cluster.clu_rep_label = clu_rep_label
                
                clu_title_info = 'cluid: {}, label distrb: {}, rep_label: {}, is event score: {}\n'.\
                    format(cluid, cluid2lbnum, str(clu_rep_label), cluster.get_event_score())
                
                for twh in clu_twarr:
                    twh['detected'] = (twh[tk.key_event_label] == clu_rep_label)
                
                det_lb_txt = [(tw['detected'], tw[tk.key_event_label], tw[tk.key_text]) for tw in clu_twarr]
                dlt_sort = sorted(det_lb_txt, key=lambda item: item[1])
                text_info = '\n'.join(['  {:<2} {:<5} {}'.format(*item) for item in dlt_sort])
                
                score_info_arr.append(clu_title_info + sep_pattern)
                text_info_arr.append(clu_title_info + text_info + sep_pattern)
            
            # all_clu_vec_arr = [cludict[cluid].get_pos_mean_vector() for cluid in cluidarr]
            # sim_matrix = au.cosine_similarity(all_clu_vec_arr)
            # cluid2top_sim = dict()
            # for idx, cluid in enumerate(cluidarr):
            #     cluster = cludict[cluid]
            #     sim_arr =
            
            fu.write_lines(textinfo_pattern.format(batch_idx), text_info_arr)
            fu.write_lines(scoreinfo_pattern.format(batch_idx), score_info_arr)
            
            oldest_twharr = batch_window.pop(0)
            for twh in oldest_twharr:
                twh.update_cluster(None)


if __name__ == '__main__':
    """ load data """
    tw_batches = get_tw_batches()
    cluid_batches = fu.load_array('cluid_evo.txt')
    
    tmu.check_time()
    twarr = au.merge_array(tw_batches)
    tu.twarr_nlp(twarr)
    tmu.check_time(print_func=lambda dt: print('{} tweets spacy over in {} s'.format(len(twarr), dt)))
    
    a = Analyzer(hold_batch_num)
    a.set_batches(tw_batches, cluid_batches)
    a.start_iter()

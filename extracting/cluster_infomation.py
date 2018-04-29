from extracting.geo_and_time.extract_time_info import get_sutime
from extracting.geo_and_time.extract_geo_location import extract_geo_freq_list
from extracting.geo_and_time.extract_time_info import get_text_time, get_event_time
from extracting.hot_and_level.event_hot import HotDegree, AttackHot
from extracting.hot_and_level.event_level import AttackLevel
from extracting.keyword_info.autophrase_multiprocess import get_quality_autophrase
from extracting.keyword_info.n_gram_keyword import get_quality_n_gram

from classifying.terror.classifier_terror import ClassifierTerror
import utils.tweet_keys as tk
import utils.spacy_utils as su
import utils.array_utils as au

from collections import OrderedDict as Od
import numpy as np
import sys
import os


class ClusterInfoGetter:
    def __init__(self):
        self.ah = AttackHot()
        self.al = AttackLevel()
        self.clf = ClassifierTerror()
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            self.sutime_obj = get_sutime()
            sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def input_cluid_twarr(self, cluid, twarr):
        txtarr = [tw[tk.key_text] for tw in twarr]
        docarr = su.textarr_nlp(txtarr)
        """ summary info """
        prob = self.clf.predict_mean_proba(twarr)
        hot = self.ah.hot_degree(twarr)
        level = self.al.get_level(twarr)
        """ geo info """
        geo_freq_list = extract_geo_freq_list(twarr, docarr)
        geo_list = self.get_readable_geo_list(geo_freq_list)
        """ time info """
        top_geo, top_freq = geo_freq_list[0] if len(geo_freq_list) > 0 else (None, None)
        text_times, utc_time = get_text_time(twarr, top_geo, self.sutime_obj)
        earliest_time, latest_time = get_event_time(twarr)
        most_time_str = utc_time.isoformat()
        earliest_time_str = earliest_time.isoformat()
        latest_time_str = latest_time.isoformat()
        time_list = [most_time_str, earliest_time_str, latest_time_str]
        """ keyword info """
        tokensarr, gram_keywords = get_quality_n_gram(txtarr, n_range=range(1, 5), len_thres=4)
        # auto_keywords = get_quality_autophrase(pidx, txtarr, conf_thres=0.5, len_thres=2)
        keywords = gram_keywords[:100]
        idxes, keywords = au.group_and_reduce(keywords, dist_thres=0.3, process_num=0)
        keywords = keywords[:50]
        """ sort twarr """
        post_twarr = self.importance_sort(twarr, docarr, tokensarr, keywords)
        post_twarr = self.clear_twarr_fields(post_twarr)
        return cluid, post_twarr, prob, hot, level, geo_list, time_list, keywords
    
    @staticmethod
    def get_readable_geo_list(geo_freq_list):
        readable_info = list()
        for geo, freq in geo_freq_list:
            # type(geo) is GoogleResult
            if geo.quality == 'locality':
                info = [geo.quality, geo.city, geo.country_long, geo.bbox, freq]
            else:
                info = [geo.quality, geo.address, geo.country_long, geo.bbox, freq]
            readable_info.append(info)
        return readable_info
    
    @staticmethod
    def importance_sort(twarr, docarr, tokensarr, keywords):
        assert len(twarr) == len(docarr) == len(tokensarr)
        exception_substitute = np.zeros((len(twarr),))
        with np.errstate(divide='raise'):
            try:
                word_num = np.array([len(tokens) for tokens in tokensarr])
                text_len = np.array([len(tw[tk.key_text]) for tw in twarr])
                len_score = (np.log(word_num) + np.log(text_len)) * 0.2
            except:
                print('len_score error')
                len_score = exception_substitute
            try:
                hd = HotDegree()
                influ_score = np.array(
                    [hd.tw_propagation(tw) + hd.user_influence(tw[tk.key_user]) for tw in twarr])
                influ_score = np.log(influ_score + 1) * 0.4
            except:
                print('influ_score error')
                influ_score = exception_substitute
        
        ents_list = [[e for e in doc.ents if len(e.text) > 1] for doc in docarr]
        geo_num_list = [sum([(e.label_ in su.LABEL_LOCATION) for e in ents]) for ents in ents_list]
        gpe_score = np.array(geo_num_list) * 0.5
        
        keywords_set = set(keywords)
        key_common_num = [len(set(tokens).intersection(keywords_set)) for tokens in tokensarr]
        keyword_score = np.array(key_common_num)
        
        score_arr = gpe_score + len_score + influ_score + keyword_score
        sorted_idx = np.argsort(score_arr)[::-1]
        # # del
        # for idx in sorted_idx[:10]:
        #     text = twarr[idx][tk.key_text]
        #     g_score = round(gpe_score[idx], 4)
        #     l_score = round(len_score[idx], 4)
        #     i_score = round(influence_score[idx], 4)
        #     sum_score = round(g_score + l_score + i_score, 4)
        #     print('{}, {}, {}, {}\n{}\n'.format(g_score, l_score, i_score, sum_score, text))
        # # del
        return [twarr[idx] for idx in sorted_idx]
    
    @staticmethod
    def clear_twarr_fields(twarr):
        do_not_copy = {tk.key_text, tk.key_orgntext, tk.key_spacy, tk.key_event_cluid, tk.key_event_label}
        post_twarr = list()
        for tw in twarr:
            new_tw = dict()
            for key in set(tw.keys()).difference(do_not_copy):
                new_tw[key] = tw[key]
            new_tw[tk.key_text] = tw[tk.key_orgntext]
            new_tw['clean_text'] = tw[tk.key_text]
            post_twarr.append(new_tw)
        return post_twarr


class ClusterInfoCarrier:
    def __init__(self, cluid, twarr, prob, hot, level, geo_list, time_list, keywords):
        self.cluid = cluid
        self.twarr = twarr
        self.prob = prob
        self.hot = hot
        self.level = level
        self.geo_list = geo_list
        self.time_list = time_list
        self.keywords = keywords
        
        self.od = Od()
        summary = Od(zip(
            ['cluster_id', 'prob', 'level', 'hot', 'keywords'], [cluid, prob, level, hot, keywords],
        ))
        geo_table = [Od(zip(['quality', 'address', 'country', 'bbox'], geo)) for geo in geo_list]
        time_table = Od(zip(['most_possible_time', 'earliest_time', 'latest_time'], time_list))
        self.od.update(zip(
            ['summary', 'geo_infer', 'time_infer', 'tweet_list'], [summary, geo_table, time_table, twarr],
        ))
    
    def transfer_od(self):
        return self.od


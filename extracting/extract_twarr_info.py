import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.tweet_utils as tu
import utils.timer_utils as tmu

from extracting.hot_and_level.event_hot import HotDegree, AttackHot
from extracting.hot_and_level.event_level import AttackLevel
from extracting.geo_and_time.extract_geo_loction import extract_sorted_readable_geo, get_loc_freq_list

import numpy as np
from prettytable import PrettyTable


hd = HotDegree()
attack_hot = AttackHot()
attack_level = AttackLevel()


def extract_twarr_full_info(twarr):
    # Assume that twarr has been processed by spacy
    docarr = [tw.get(tk.key_spacy) for tw in twarr]
    loc_freq_list = get_loc_freq_list(docarr)
    top_geo_info = extract_sorted_readable_geo(twarr, docarr)
    hot = attack_hot.hot_degree(twarr)
    level = attack_level.get_level(twarr)
    sorted_twarr = importance_sort(twarr, docarr)
    return [loc_freq_list, top_geo_info, hot, level, sorted_twarr]


def importance_sort(twarr, docarr):
    exception_substitute = np.array([0] * len(twarr))
    try:
        gpe_num = [sum([1 for e in d.ents if e.label_ == su.LABEL_GPE and len(e.text) > 1]) for d in docarr]
        gpe_score = np.array(gpe_num) * 0.5
    except:
        gpe_score = exception_substitute
    try:
        word_num = np.array([len(doc) for doc in docarr])
        text_len = np.array([len(tw[tk.key_text]) for tw in twarr])
        len_score = np.log(word_num + text_len) * 0.2
    except:
        len_score = exception_substitute
    try:
        influence_score = np.array([hd.tw_propagation(tw) + hd.user_influence(tw[tk.key_user]) for tw in twarr])
        influence_score = np.log(influence_score) * 0.4
    except:
        influence_score = exception_substitute
    
    score_arr = gpe_score + len_score + influence_score
    sorted_idx = np.argsort(score_arr)[::-1]
    
    for idx in sorted_idx[:10]:
        text = twarr[idx][tk.key_text]
        g_score = round(gpe_score[idx], 4)
        l_score = round(len_score[idx], 4)
        i_score = round(influence_score[idx], 4)
        sum_score = round( g_score + l_score + i_score, 4)
        print('{}, {}, {}, {}\n{}\n'.format(g_score, l_score, i_score, sum_score, text))
    
    return [twarr[idx] for idx in sorted_idx]


if __name__ == '__main__':
    base = '/home/nfs/yangl/event_detection/testdata/event_corpus/'
    event_files = fi.listchildren(base, fi.TYPE_FILE, pattern='.txt$', concat=True)
    for file in event_files:
        twarr = fu.load_array(file)
        if len(twarr) > 50:
            continue
        print(file)
        tu.twarr_nlp(twarr)
        
        top_geo_info, hot, level, sorted_twarr = extract_twarr_full_info(twarr)
        
        def p(g_info):
            table = PrettyTable(["种类", "地址", "国家", "坐标范围", "频次"])
            for info in g_info:
                table.add_row(info)
            print(table, '\n')
        p(top_geo_info)
        
        for idx, tw in enumerate(twarr):
            print(idx, tw[tk.key_text])
        print('\n\n\n')

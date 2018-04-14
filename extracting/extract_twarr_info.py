import re
import json

import utils.file_iterator as fi
import utils.function_utils as fu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.tweet_utils as tu

from extracting.keyword_info.autophrase_multiprocess import autophrase_multi_top_results
from extracting.hot_and_level.event_hot import HotDegree, AttackHot
from extracting.hot_and_level.event_level import AttackLevel
from extracting.geo_and_time.extract_geo_location import get_readable_from_geo_freq_list, extract_sorted_geo_freq_list
from extracting.geo_and_time.extract_time_info import get_text_time, get_event_time

import numpy as np
from prettytable import PrettyTable
from collections import OrderedDict


hd = HotDegree()
attack_hot = AttackHot()
attack_level = AttackLevel()


"""
每个聚类使用一个json字符串表示，对应的对象含如下字段

= 'summary': dict, 概括聚类基本信息，含如下字段
    - 'cluster_id': int, 区分聚类，一个id只分配给一个聚类，即使聚类消失，id也不会给其他聚类
    - 'level': str, 表明事件等级，目前定义了4个级别(一般事件、较大事件、重大事件、特别重大事件)
    - 'hot': int, 表明事件热度，最大值不定，参考用
    - 'keywords': list, 每个元素类型为str，代表一个关键词(组)，元素按重要程度从高到低排序

= 'inferred_geo': list, 聚类的地点提取结果，元素按确定程度从高到低排序
    每个元素类型为dict，含如下字段
        - 'quality': str, 表明该地点行政级别，
        - 'address': str, 表明该地点所处位置，行政级别为'city'时该字段
        - 'country': str, 表明该地点所属的国家全称
        - 'bbox': dict, 表明该地点的坐标范围(矩形)，含如下字段
            + 'northeast': list, 该地点的东北坐标，[纬度，经度]
            + 'southwest': list, 该地点的西南坐标，[纬度，经度]
        # - 'weight': float, 表明该地点在聚类中被识别的相对权重

= 'inferred_time': dict, 聚类的时间提取结果，含如下字段
    - 'most_possible_time': str, ISO-8601格式，表明从聚类推断的最可能指向的时间(段)
    - 'earliest_time': str, ISO-8601格式，表明聚类中发布最早的推文的时间
    - 'latest_time': str, ISO-8601格式，表明聚类中发布最晚的推文的时间

= 'tweet_list': list, 聚类的推文列表，元素按重要程度从高到低排序
    每个元素类型为str，即推特的json格式字符串
"""


def cluster_list2json_list(cludict):
    def is_valid_cluster(cluster):
        return cluster.twnum > 30
    cluid_list = sorted([cluid for cluid in cludict.keys() if is_valid_cluster(cludict[cluid])])
    cluster_list = [cludict[cluid] for cluid in cluid_list]
    twarr_list = [cluster.get_twarr() for cluster in cluster_list]
    keywords_list = autophrase_multi_top_results(twarr_list, process_num=10, max_word_num=20)
    args = list(zip(cluid_list, twarr_list, keywords_list))
    dict_list = [cluster2od(*arg) for arg in args]
    json_list = [json.dumps(od) for od in dict_list]
    return json_list


def cluster2od(cluid, clu_twarr, keywords):
    # Assume that twarr of cluster has been processed by spaCy
    hot, level, geo_info_list, time_info_list, sorted_twarr = extract_twarr_full_info(clu_twarr)
    
    summary = OrderedDict(zip(['cluster_id', 'level', 'hot', 'keywords'], [cluid, level, hot, keywords]))
    inferred_geo = [OrderedDict(zip(['quality', 'address', 'country', 'bbox'], geo)) for geo in geo_info_list]
    inferred_time = OrderedDict(zip(['most_possible_time', 'earliest_time', 'latest_time'], time_info_list))
    tweet_list = post_process_twarr(sorted_twarr)
    
    od = OrderedDict(zip(
        ['summary', 'inferred_geo', 'inferred_time', 'tweet_list'],
        [summary, inferred_geo, inferred_time, tweet_list]
    ))
    from pprint import pprint
    pprint(od)
    return od


def extract_twarr_full_info(twarr):
    # Assume that twarr has been processed by spaCy
    tu.twarr_nlp(twarr)
    docarr = [tw[tk.key_spacy] for tw in twarr]
    geo_freq_list = extract_sorted_geo_freq_list(twarr, docarr)
    geo_info_list = get_readable_from_geo_freq_list(geo_freq_list)
    
    top_geo = geo_freq_list[0][0] if len(geo_freq_list) > 0 else None
    text_times, utc_time = get_text_time(twarr, top_geo)
    earliest_time, latest_time = get_event_time(twarr)
    most_time_str = utc_time.isoformat()
    earliest_time_str = earliest_time.isoformat()
    latest_time_str = latest_time.isoformat()
    time_info_list = [most_time_str, earliest_time_str, latest_time_str]
    
    hot = attack_hot.hot_degree(twarr)
    level = attack_level.get_level(twarr)
    sorted_twarr = importance_sort(twarr, docarr)
    return hot, level, geo_info_list, time_info_list, sorted_twarr


def post_process_twarr(twarr):
    """ Only reserve fields of each tw that can be converted by json """
    json_arr = list()
    do_not_copy = {tk.key_text, tk.key_orgntext, tk.key_spacy, tk.key_event_cluid}
    for tw in twarr:
        post_tw = dict()
        for key in set(tw.keys()).difference(do_not_copy):
            post_tw[key] = tw[key]
        post_tw[tk.key_text] = tw[tk.key_orgntext]
        json_arr.append(json.dumps(post_tw))
    return json_arr


def importance_sort(twarr, docarr):
    assert len(twarr) == len(docarr)
    exception_substitute = np.zeros([len(twarr), ])
    try:
        gpe_num = [sum([1 for e in d.ents if e.label_ == su.LABEL_GPE and len(e.text) > 1]) for d in docarr]
        gpe_score = np.array(gpe_num) * 0.5
    except:
        gpe_score = exception_substitute
    try:
        word_num = np.array([len(doc) for doc in docarr])
        text_len = np.array([len(tw[tk.key_text]) for tw in twarr])
        len_score = (np.log(word_num) + np.log(text_len)) * 0.2
    except:
        len_score = exception_substitute
    try:
        influence_score = np.array([hd.tw_propagation(tw) + hd.user_influence(tw[tk.key_user]) for tw in twarr])
        influence_score = np.log(influence_score) * 0.4
    except:
        influence_score = exception_substitute
    score_arr = gpe_score + len_score + influence_score
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


if __name__ == '__main__':
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/'
    files = fi.listchildren(base, fi.TYPE_FILE, concat=True)
    for file in files:
        twarr = fu.load_array(file)
        if len(twarr) > 50:
            continue
        print(file)
        tu.twarr_nlp(twarr)
        
        hot, level, geo_info_list, time_info_list, sorted_twarr = extract_twarr_full_info(twarr)
        
        table = PrettyTable(["种类", "地址", "国家", "坐标范围", "频次"])
        for info in geo_info_list:
            table.add_row(info)
        print(table, '\n')
        
        # table = PrettyTable(["推测时间", "推文文本", "时间词", "推文创建时间", "utc_offset"])
        # # table.padding_width = 1
        # for time in text_times:
        #     table.add_row(time)
        # print(table, '\n')
        break

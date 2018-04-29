import re
import json

import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.tweet_utils as tu

from classifying.terror.classifier_terror import ClassifierTerror
from extracting.keyword_info.autophrase_multiprocess import autophrase_multi_top_results
from extracting.hot_and_level.event_hot import HotDegree, AttackHot
from extracting.hot_and_level.event_level import AttackLevel
from extracting.geo_and_time.extract_geo_location import get_readable_geo_list, extract_geo_freq_list
from extracting.geo_and_time.extract_time_info import get_text_time, get_event_time
from extracting.keyword_info.n_gram_keyword import get_quality_keywords, valid_tokens_of_text

from pprint import pprint
import numpy as np
from collections import OrderedDict


hd = HotDegree()
attack_hot = AttackHot()
attack_level = AttackLevel()


"""
每个聚类使用一个json字符串表示，对应的对象含如下字段

= 'summary': dict, 概括聚类基本信息，含如下字段
    - 'cluster_id': int, 区分聚类，一个id只分配给一个聚类，即使聚类消失，id也不会给其他聚类
    - 'prob': float, 聚类置信度，标志聚类的预警程度
    - 'level': str, 表明事件等级，目前定义了4个级别(一般事件、较大事件、重大事件、特别重大事件)
    - 'hot': int, 表明事件热度，取值范围(0, 100)
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


class ClfHolder:
    terror_clf = None


def get_clf():
    if ClfHolder.terror_clf is None:
        ClfHolder.terror_clf = ClassifierTerror()
    return ClfHolder.terror_clf


def dump_cludict_into_path(path, cluid_twarr_list):
    if len(cluid_twarr_list) < 1:
        return
    print("accept {} clusters".format(len(cluid_twarr_list)))
    cluid_od_list = cluid_twarr_list2cluid_od_list(cluid_twarr_list)
    for cluid, od in cluid_od_list:
        json_file = fi.join(path, '{}.json'.format(cluid))
        json.dump(od, open(json_file, 'w', encoding='utf8'))
# --del--
        readable_od(fi.join(path, '{}_r.txt'.format(cluid)), od)
def readable_od(file, od):
    str_list = list()
    summary_attr_list = ['cluster_id', 'prob', 'level', 'hot', 'keywords']
    for attr in summary_attr_list:
        str_list.append("{}: {}".format(attr, od['summary'][attr]))
    str_list.append('\n')
    
    columns = ['quality', 'address', 'country', 'bbox']
    str_list.append('inferred_geo')
    pattern = '\t\t\t'.join(['{}']*len(columns))
    str_list.append(pattern.format(*columns))
    for geo in od['inferred_geo']:
        str_list.append(pattern.format(*geo.values()))
    str_list.append('\n')
    
    str_list.append('inferred_time')
    time_attr_list = ['most_possible_time', 'earliest_time', 'latest_time']
    for attr in time_attr_list:
        str_list.append("{}: {}".format(attr, od['inferred_time'][attr]))
    str_list.append('\n')
    
    twarr = od['tweet_list']
    for tw in twarr:
        str_list.append(tw)
    
    fu.write_lines(file, str_list)
# --del--


def cluid_twarr_list2cluid_od_list(cluid_twarr_list):
    cluid_list, twarr_list = list(zip(*cluid_twarr_list))
    keywords_list = autophrase_multi_top_results(twarr_list, process_num=10, max_word_num=30)
    # keywords_list = [[] for _ in range(len(twarr_list))]
    cluid_od_list = [cluster2cluid_od(*args) for args in list(zip(cluid_list, twarr_list, keywords_list))]
    return cluid_od_list


def cluster2cluid_od(cluid, clu_twarr, auto_keywords):
    # Assume that twarr of cluster has been processed by spaCy
    docarr = [tw[tk.key_spacy] for tw in clu_twarr]
    prob, my_keywords, hot, level, geo_list, time_list, sorted_twarr = extract_twarr_full_info(clu_twarr, docarr)
    summary = OrderedDict(zip(
        ['cluster_id', 'prob', 'level', 'hot', 'keywords'],
        [cluid, prob, level, hot, my_keywords + auto_keywords],
    ))
    inferred_geo = [OrderedDict(zip(['quality', 'address', 'country', 'bbox'], geo)) for geo in geo_list]
    inferred_time = OrderedDict(zip(['most_possible_time', 'earliest_time', 'latest_time'], time_list))
    tweet_list = twarr2json_string_array(sorted_twarr)
    od = OrderedDict(zip(
        ['summary', 'inferred_geo', 'inferred_time', 'tweet_list'],
        [summary, inferred_geo, inferred_time, tweet_list],
    ))
    return cluid, od


def extract_twarr_full_info(twarr, docarr):
    geo_freq_list = extract_geo_freq_list(twarr, docarr)
    geo_info_list = get_readable_geo_list(geo_freq_list)
    
    top_geo = geo_freq_list[0][0] if len(geo_freq_list) > 0 else None
    text_times, utc_time = get_text_time(twarr, top_geo)
    earliest_time, latest_time = get_event_time(twarr)
    earliest_time_str = earliest_time.isoformat()
    latest_time_str = latest_time.isoformat()
    most_time_str = utc_time.isoformat()
    time_info_list = [most_time_str, earliest_time_str, latest_time_str]
    
    prob = get_clf().predict_mean_proba(twarr)
    hot = attack_hot.hot_degree(twarr)
    level = attack_level.get_level(twarr)
    sorted_twarr, keywords = importance_sort(twarr, docarr)
    return prob, keywords, hot, level, geo_info_list, time_info_list, sorted_twarr


def importance_sort(twarr, docarr):
    exception_substitute = np.zeros([len(twarr), ])
    with np.errstate(divide='raise'):
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
            influ_score = np.array([hd.tw_propagation(tw) + hd.user_influence(tw[tk.key_user]) for tw in twarr])
            influ_score = np.log(influ_score + 1) * 0.4
        except:
            influ_score = exception_substitute
    
    tokens_list = [valid_tokens_of_text(tw[tk.key_text].lower()) for tw in twarr]
    keywords = get_quality_keywords(tokens_list, n_range=4, threshold=0.4, top_k=40)
    keyword_set = set(keywords)
    keyword_score = np.array([len(set(tokens).intersection(keyword_set)) for tokens in tokens_list])
    
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
    return [twarr[idx] for idx in sorted_idx], keywords


def twarr2json_string_array(twarr):
    """ Only reserve fields of each tw that can be converted by json """
    json_str_arr = list()
    do_not_copy = {tk.key_text, tk.key_orgntext, tk.key_spacy, tk.key_event_cluid, tk.key_event_label}
    for tw in twarr:
        post_tw = dict()
        for key in set(tw.keys()).difference(do_not_copy):
            post_tw[key] = tw[key]
        post_tw[tk.key_text] = tw[tk.key_orgntext]
        # TODO json_str_arr.append(json.dumps(post_tw))
        json_str_arr.append(re.sub('\n', '.', tw[tk.key_orgntext]))
    return json_str_arr


if __name__ == '__main__':
    from collections import Counter
    import utils.pattern_utils as pu
    file = "/home/nfs/cdong/tw/src/calling/temp_outputs/2885.json"
    textarr = fu.load_array(file)[0]["tweet_list"]
    docarr = su.textarr_nlp(textarr)
    pos_keys = {su.pos_prop, su.pos_comm, su.pos_verb}
    counter = Counter()
    for doc in docarr:
        for token in doc:
            word, pos = token.text.strip().lower(), token.pos_
            if pos in pos_keys and word not in pu.stop_words:
                counter[word] += 1
    print(counter.most_common(40))
    # print(len(twarr))
    # pprint(obj, compact=True)
    exit()
    
    import utils.timer_utils as tmu
    
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive'
    files = fi.listchildren(base, fi.TYPE_FILE, concat=True)
    tmu.check_time()
    _twarr_list = list()
    for file in files:
        _twarr = fu.load_array(file)
        if len(_twarr) < 50:
            continue
        tu.twarr_nlp(_twarr)
        _twarr_list.append(_twarr)
    print('{}/ {}'.format(len(_twarr_list), len(files)))
    tmu.check_time()
    
    for _idx, _twarr in enumerate(_twarr_list):
        _od = cluster2cluid_od(_idx, _twarr, ["not now", "ta"])
        _od.pop('tweet_list')
        pprint(_od)
        # fu.write_lines(str(_idx), [json.dumps(_od)])
    tmu.check_time()
    
    # hot, level, geo_info_list, time_info_list, sorted_twarr = extract_twarr_full_info(twarr)
    #
    # table = PrettyTable(["种类", "地址", "国家", "坐标范围", "频次"])
    # for info in geo_info_list:
    #     table.add_row(info)
    # print(table, '\n')
    #
    # # table = PrettyTable(["推测时间", "推文文本", "时间词", "推文创建时间", "utc_offset"])
    # # # table.padding_width = 1
    # # for time in text_times:
    # #     table.add_row(time)
    # # print(table, '\n')
    # break

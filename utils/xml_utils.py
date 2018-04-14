from collections import OrderedDict
import xml.dom.minidom as minidom

from pprint import pprint
import pandas as pd
import xmltodict

import utils.function_utils as fu
import json


def parse_cluster_to_ordereddict(cluster, twarr_info):
    # cluid = cluster_info
    # readable_info_list, text_times, earliest_time_str, latest_time_str, hot, level, sorted_twarr = twarr_info
    od = OrderedDict()
    
    clu_info = OrderedDict()
    clu_info["id"] = 124386342
    clu_info["level"] = "{}({})".format(2, 'just soso')
    clu_info["hot"] = 1234
    
    geo_list = [
        ["japan", "12", "32"],
        ["bangkok", "523", "435"],
        ["italy", "1234", "431"],
    ]
    geo_infer = array2ordereddict(geo_list, ['name', 'lat', 'lng'], "geo_")
    
    time_list = [
        ["20160234 12:23:73", "nimabi"],
        ["20179212 32:56:89", "tomorrow"],
    ]
    time_infer = OrderedDict()
    time_infer["earliest_time"] = "20180902 12:23:21"
    time_infer["latest_time"] = "20180902 12:23:21"
    time_text = array2ordereddict(time_list, ["inferred", "text"], "time_")
    time_infer.update(time_text)
    
    tw_file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/2016-01-29_attack_Dalori.json"
    twarr = fu.load_array(tw_file)[:10]
    jsonarr = [[json.dumps(tw)[50:70]] for tw in twarr]
    tweet_list = array2ordereddict(jsonarr, row_prefix="tweet_")
    
    od['cluster_info'] = clu_info
    od['inferred_geo'] = geo_infer
    od['inferred_time'] = time_infer
    od['sorted_twarr'] = tweet_list
    od = {'cluster': od}
    return od


def array2ordereddict(item_array, columns=None, row_prefix=''):
    table = pd.Series() if columns is None else pd.DataFrame(columns=columns)
    for idx, item in enumerate(item_array):
        table.loc["{}{}".format(row_prefix, idx)] = item
    res_ordereddict = table.T.to_dict(into=OrderedDict)
    return res_ordereddict


def parse_xml_string_to_dict(xml_str):
    output_dict = xmltodict.parse(xml_str)
    return output_dict


def parse_dict_to_xml_string(input_dict):
    xml_str = xmltodict.unparse(input_dict)
    return xml_str


def dump_dict(output_file, input_dict, mode='w'):
    xml_str = parse_dict_to_xml_string(input_dict)
    dump_xml_string(output_file, xml_str, mode)


def dump_xml_string(output_file, xml_str, mode='w'):
    dom = minidom.parseString(xml_str)
    dom.writexml(open(output_file, mode), indent='', addindent='    ', newl='\n', encoding="utf-8")


if __name__ == '__main__':
    out_file = "test_out.xml"
    # clu_d = parse_cluster_to_ordereddict(None, None)
    # dump_dict(out_file, clu_d)
    d = parse_xml_string_to_dict(open(out_file).read().encode())
    # pprint(d)
    import json
    print(json.dumps(d))
    exit()

import json
import operator
from collections import Counter

import sqlite3
from prettytable import PrettyTable
from geocoder.google import GoogleResult


conn = sqlite3.connect('/home/nfs/cdong/tw/src/models/geo/geo_location')
c = conn.cursor()


def capitalize(text):
    return ' '.join([word.capitalize() for word in text.split()])


def geo_location_extract(doc_array):
    loc_freq_counter = Counter()
    for doc in doc_array:
        ents = Counter([capitalize(ent.text) for ent in doc.ents if (ent.label_ == 'GPE' and len(ent.text) > 1)])
        loc_freq_counter += ents
    return loc_freq_counter.most_common()


def get_geolocation_of_array(loc_freq_list, coordinates):
    # loc_freq_list = [('place1', 10), ('place2', 7), ('place3', 2), ...]
    if len(loc_freq_list) == 0:
        return
    # get geolocation for all location > 1/3 first location matches
    geo_number = 1
    top_loc, top_freq = loc_freq_list[0]
    for loc, freq in loc_freq_list[1:]:
        if freq >= top_freq / 3:
            geo_number += 1
    geo_list = _get_geo_lists(loc_freq_list[:geo_number])
    geo_list = reevaluate_with_coordinates(geo_list, coordinates)
    geo_list = _determine_index_order(geo_list)
    print('new array:')
    table = PrettyTable(["种类", "地址", "国家", "坐标范围", "精确度"])
    for idx, geo in enumerate(geo_list):
        if idx == 0:
            text = '推测地点'
        else:
            text = '其他可能地点'
        _show_geo(geo, text, table)
    print(table)
    print('\n\n')


def _get_geo_lists(loc_freq_list):
    geo_list = list()
    for loc, freq in loc_freq_list:
        geo = _get_geo_location(loc)
        if geo is None:
            continue
        else:
            geo_list.append((geo, freq))
    return geo_list


def _get_geo_location(loc_name):
    geo_info = _get_geo_result(loc_name)
    return None if geo_info is None else GoogleResult(geo_info)


def _get_geo_result(loc_name):
    c.execute("SELECT i2g.GEO FROM nameToId AS n2i, idToGeo AS i2g WHERE n2i.NAME=:place AND n2i.ID=i2g.ID",
              {"place": loc_name})
    geo_result = c.fetchone()
    if geo_result is None:
        return None
    geo_info = json.loads(geo_result[0])
    print("get {} from database".format(geo_info))
    return geo_info


def reevaluate_with_coordinates(geo_freq_list, coordinate_list):
    if not geo_freq_list:
        return list()
    """ for each geo, count the coordinates that falls in its region """
    coordinates_match = list()
    for idx, (geo, freq) in enumerate(geo_freq_list):
        match_num = 0
        for coord_lng, coord_lat in coordinate_list:
            if geo.west < coord_lng < geo.east and geo.south < coord_lat < geo.north:
                match_num += 1
        coordinates_match.append(match_num)
    origin_max_match = geo_freq_list[0][1]
    coord_match_max = max(coordinates_match)
    if coord_match_max == 0:
        return geo_freq_list
    new_geo_dict = dict()
    """ 0.42 is approximately 3/7 """
    for idx, (geo, freq) in enumerate(geo_freq_list):
        new_geo_dict[geo] = freq + 0.42 * origin_max_match * coordinates_match[idx] / coord_match_max
    new_geo_list = sorted(new_geo_dict.items(), key=operator.itemgetter(1), reverse=True)
    print(1, new_geo_list)
    print(2, geo_freq_list)
    return new_geo_list


def _show_geo(geo, text, table):
    geo_loc = geo[0]
    geo_num = geo[1]
    if geo_loc.quality == 'country':
        table.add_row(
            [text, geo_loc.address + '({})'.format(geo_num), geo_loc.country_long, geo_loc.bbox,
             geo_loc.quality])
    elif geo_loc.quality == 'locality':
        table.add_row(
            [text, geo_loc.city + '({})'.format(geo_num), geo_loc.country_long, geo_loc.bbox,
             geo_loc.quality])
    else:
        table.add_row(
            [text, geo_loc.address + '({})'.format(geo_num), geo_loc.country_long, geo_loc.bbox,
             geo_loc.quality])


def _determine_index_order(geo_list, index1=0, index2=1):
    if index1 >= len(geo_list) or index2 >= len(geo_list) or index1 == index2:
        return geo_list
    geo1_quality = geo_list[index1][0].quality
    geo2_quality = geo_list[index2][0].quality
    if geo1_quality == 'country':
        if geo2_quality != 'country':
            temp = geo_list[index1]
            geo_list[index1] = geo_list[index2]
            geo_list[index2] = temp
            return geo_list
        else:
            index2 += 1
            return _determine_index_order(geo_list, index1, index2)
    else:
        return geo_list


if __name__ == '__main__':
    sql_create_table_name_to_id = """ CREATE TABLE IF NOT EXISTS nameToId (NAME text PRIMARY KEY,ID text);"""
    sql_create_table_id_to_Geo = """ CREATE TABLE IF NOT EXISTS idToGeo (ID text PRIMARY KEY,GEO text NOT NULL); """
    
    if conn:
        print('Geo database opened successfully')
    c.execute(sql_create_table_name_to_id)
    c.execute(sql_create_table_id_to_Geo)
    conn.commit()

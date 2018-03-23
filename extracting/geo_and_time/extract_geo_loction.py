import json
import operator
from collections import Counter

import utils.tweet_keys as tk
import utils.pattern_utils as pu

import sqlite3
from geocoder.google import GoogleResult


conn = sqlite3.connect('/home/nfs/cdong/tw/src/models/geo/geo_location')
c = conn.cursor()


def extract_sorted_readable_geo(twarr, docarr):
    coordinates = get_coordinates(twarr)
    loc_freq_list = get_loc_freq_list(docarr)
    geo_freq_list = get_geo_freq_list(loc_freq_list, coordinates)
    readable_info_list = extract_info_from_geo_freq_list(geo_freq_list)
    return readable_info_list


def get_loc_freq_list(docarr):
    loc_freq_counter = Counter()
    for doc in docarr:
        ents = [pu.capitalize(e.text) for e in doc.ents if (e.label_ == 'GPE' and len(e.text) > 1)]
        loc_freq_counter += Counter(ents)
    if len(loc_freq_counter) == 0:
        return list()
    loc_freq_list = loc_freq_counter.most_common()
    top_loc, top_freq = loc_freq_list[0]
    geo_valid_num = sum([freq >= top_freq / 4 for loc, freq in loc_freq_list[1:]]) + 1
    loc_freq_list = loc_freq_list[:geo_valid_num]
    return loc_freq_list


def get_geo_freq_list(loc_freq_list, coordinates):
    geo_freq_list = _get_geo_freq_list(loc_freq_list)
    geo_freq_list = _reevaluate_with_coordinates(geo_freq_list, coordinates)
    geo_freq_list = _determine_index_order(geo_freq_list)
    return geo_freq_list


def extract_info_from_geo_freq_list(geo_freq_list):
    geo_info = list()
    for geo, freq in geo_freq_list:
        if geo.quality == 'country':
            info = [geo.quality, geo.address, geo.country_long, geo.bbox, freq]
        elif geo.quality == 'locality':
            info = [geo.quality, geo.city, geo.country_long, geo.bbox, freq]
        else:
            info = [geo.quality, geo.address, geo.country_long, geo.bbox, freq]
        geo_info.append(info)
    return geo_info


def get_coordinates(twarr):
    k_coord = tk.key_coordinates
    return [tw[k_coord][k_coord] for tw in twarr if (k_coord in tw and tw[k_coord] is not None)]


def _get_geo_freq_list(loc_freq_list):
    geo_list = list()
    for loc, freq in loc_freq_list:
        geo = get_geo_by_loc_name(loc)
        if geo is None:
            continue
        else:
            geo_list.append((geo, freq))
    return geo_list


def get_geo_by_loc_name(loc_name):
    c.execute("SELECT i2g.GEO FROM nameToId AS n2i, idToGeo AS i2g WHERE n2i.NAME=:place AND n2i.ID=i2g.ID",
              {"place": loc_name})
    geo_result = c.fetchone()
    if geo_result is None:
        return None
    else:
        return GoogleResult(json.loads(geo_result[0]))


def _reevaluate_with_coordinates(geo_freq_list, coordinate_list):
    if not geo_freq_list:
        return list()
    # for each geo, count the coordinates that falls in its region
    coordinates_match = list()
    for idx, (geo, freq) in enumerate(geo_freq_list):
        match_num = 0
        for coord_lng, coord_lat in coordinate_list:
            if geo.west < coord_lng < geo.east and geo.south < coord_lat < geo.north:
                match_num += 1
        coordinates_match.append(match_num)
    origin_max_match = geo_freq_list[0][1]
    coord_max_match = max(coordinates_match)
    if coord_max_match == 0:
        return geo_freq_list
    new_geo_list = list()
    for idx, (geo, freq) in enumerate(geo_freq_list):
        # 0.42 is approximately 3/7
        new_freq = freq + 0.42 * origin_max_match * coordinates_match[idx] / coord_max_match
        new_geo_list.append((geo, new_freq))
    new_geo_list = sorted(new_geo_list, key=operator.itemgetter(1), reverse=True)
    return new_geo_list


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

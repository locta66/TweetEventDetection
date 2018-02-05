import json
import geocoder
from geocoder.google import GoogleResult
import socks
import socket
import time as systime
import sqlite3

socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
socket.socket = socks.socksocket

'''
get geolocation for all location > 1/3 first location matches
'''
sql_create_table_name_to_id = """ CREATE TABLE IF NOT EXISTS nameToId (NAME text PRIMARY KEY,ID text);"""
sql_create_table_id_to_Geo = """ CREATE TABLE IF NOT EXISTS idToGeo (ID text PRIMARY KEY,GEO text NOT NULL); """

conn = sqlite3.connect('D:\event_detection\project\geo\geo_location')
if conn:
    print('Geo database opened successfully')

c = conn.cursor()
c.execute(sql_create_table_name_to_id)
c.execute(sql_create_table_id_to_Geo)
conn.commit()


def get_geolocation_of_array(location_list):
    search_list = location_list[:]
    # get geolocation for all location > 1/3 first location matches
    geo_number = 1
    for loc in search_list[1:]:
        if loc[1] >= search_list[0][1] / 3:
            geo_number += 1
    geo_list = _get_geo_lists(search_list, geo_number=geo_number)
    geo_list = _detemine_geo_show_order(geo_list)
    print('new array:')
    for idx, geo in enumerate(geo_list):
        if idx == 0:
            print('推测地点:')
        else:
            print('其他可能地点')

        _show_geo(geo)
    print('\n\n\n')


def _show_geo(geo):
    geo_loc = geo[0]
    geo_num = geo[1]
    if geo_loc.quality == 'country':
        print(geo_loc.address + '({})'.format(geo_num), geo_loc.bbox, geo_loc.quality)
    elif geo_loc.quality == 'locality':
        print(geo_loc.city + '({})'.format(geo_num), geo_loc.country_long, geo_loc.bbox, geo_loc.quality)
    else:
        print(geo_loc.address + '({})'.format(geo_num), geo_loc.bbox, geo_loc.quality)


def _get_first_geo_location_from_location_list(location_list):
    if not location_list:
        return False, None, 0
    name1 = location_list[0][0]
    num_match = location_list[0][1]
    # g1 = geocoder.geonames(name1, key='yueying')
    g1 = _get_geo_result(name1)
    if not g1:
        location_list = location_list[1:]
        # return __get_first_geo_location_from_location_list(location_list)
        return False, None, num_match
    return True, g1, num_match


def _get_geo_result(name):
    c.execute("SELECT * FROM nameToId WHERE NAME =:place", {"place": name})
    id_result = c.fetchone()
    if id_result:
        geo_id = id_result[1]
        c.execute("SELECT * FROM idToGeo WHERE ID =:geoid", {"geoid": geo_id})
        geo_result = c.fetchone()
        if geo_result:
            raw = json.loads(geo_result[1])
            print("get {} from database".format(name))
            return GoogleResult(raw)

    for i in range(5):
        g1 = geocoder.google(name)
        if g1:
            geo_id = g1.raw['place_id']
            raw = json.dumps(g1.raw)
            c.execute("INSERT OR IGNORE INTO nameToId (NAME,ID) \
        VALUES (:name, :geoid)", {"name": name, "geoid": geo_id})
            c.execute("INSERT OR IGNORE INTO idToGeo (ID,GEO) \
        VALUES (:geoid,:raw)", {"geoid": geo_id, "raw": raw})
            conn.commit()
            print('insert {} success'.format(name))
            return g1
        else:
            systime.sleep(1)
    return None


'''
reorder geo_list to show city first rather than country 
just fit for google api now 

geo_list show strategy:
    exchange idx1 idx2 if idx1 is country and idx2 less then country:
        in the case idx1 idx2 is same country
    
'''


def _detemine_geo_show_order(geo_list, index1=0, index2=1):
    if index1 >= len(geo_list) or index2 >= len(geo_list) or index1 == index2:
        return geo_list
    if geo_list[index1][0].quality == 'country':
        if geo_list[index2][0].quality != 'country':
            g_index2 = geo_list[index2]
            new_geo_list = geo_list[:]
            new_geo_list.pop(index2)
            new_geo_list.insert(0, g_index2)
            return new_geo_list
        else:
            index2 += 1
            return _detemine_geo_show_order(geo_list, index1, index2)
    else:
        return geo_list


def _get_geo_lists(location_list, geo_number=3):
    geo_list = list()
    for i in range(geo_number):
        success, geo, match_num = _get_first_geo_location_from_location_list(location_list[i:])
        if not success:
            return geo_list
        else:
            geo_list.append((geo, match_num))
    return geo_list

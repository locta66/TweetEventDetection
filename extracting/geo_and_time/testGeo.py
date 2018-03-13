from extracting.geo_and_time.extract_geo_loction import get_geolocation_of_array, geo_location_extract

import utils.file_iterator as fi
import utils.function_utils as fu
import utils.tweet_keys as tk
import utils.spacy_utils as su


files_list = fi.listchildren('/home/nfs/yangl/event_detection/testdata/event_corpus', fi.TYPE_FILE, concat=True)[:1]


for f in files_list:
    print('\'\'\'\nfile :{}\n\'\'\'\n'.format(f))
    twarr = fu.load_array(f)
    textarr = [t['text'] for t in twarr]
    k_coord = tk.key_coordinates
    coordinates = [tw[k_coord][k_coord] for tw in twarr if k_coord in tw and tw[k_coord] is not None]
    print('coordinates', len(coordinates), coordinates[:5])
    
    docarr = su.textarr_nlp(textarr)
    loc_freq_list = geo_location_extract(docarr)
    get_geolocation_of_array(loc_freq_list, coordinates)

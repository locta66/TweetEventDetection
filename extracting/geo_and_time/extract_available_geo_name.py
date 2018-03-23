from collections import Counter

# import utils.multiprocess_utils as mu
# import utils.spacy_utils as su
# import utils.tweet_keys as tk
# import utils.file_iterator as fi
# import utils.pattern_utils as pu
# import utils.function_utils as fu
from utils.utils_loader import *


en_nlp = su.get_nlp('en')
process_num = 5


def extract_geo_names(file_list):
    file_list_blocks = mu.split_multi_format(file_list, process_num)
    res_list = mu.multi_process(extract_geo_names_from_files, [[flist] for flist in file_list_blocks])
    geo_names = Counter()
    for res in res_list:
        geo_names += res
    return geo_names


def extract_geo_names_from_files(file_list):
    geo_names = Counter()
    for file in file_list:
        twarr = fu.load_array(file)
        file_geo_names = extract_geo_names_from_twarr(twarr)
        geo_names += file_geo_names
    return geo_names


def extract_geo_names_from_twarr(twarr):
    textarr = [tw[tk.key_text] for tw in twarr if len(tw[tk.key_text]) > 20][:3000]
    docarr = su.textarr_nlp(textarr, nlp=en_nlp)
    geo_names = list()
    for doc in docarr:
        for ent in doc.ents:
            ent_name = ent.text.strip()
            if ent.label_ in su.LABEL_IS_LOCATION and pu.has_azAZ(ent_name) and len(ent_name) > 1:
                geo_names.append(pu.capitalize(ent_name))
    return Counter(geo_names)


def merge_geo_names():
    from extracting.geo_and_time.extract_geo_loction import get_geo_by_loc_name
    base = "/home/nfs/cdong/tw/src/extracting/geo_and_time/"
    files = fi.listchildren(base, fi.TYPE_FILE, ".json$")
    ifd = IdFreqDict()
    for file in files:
        word_freq_list = fu.load_array(file)
        for word, freq in word_freq_list:
            if len(word) <= 1:
                continue
            if len(word) == 2:
                word = word.upper()
            if get_geo_by_loc_name(word) is not None:
                continue
            ifd.count_word(word, freq)
    fu.dump_array("geo_freq.json", ifd.most_common())


if __name__ == "__main__":
    # merge_geo_names()
    # exit()
    base = "/home/nfs/cdong/tw/origin/"
    sub_files = fi.listchildren(base, fi.TYPE_FILE, concat=True)[3000:]
    geo_freq = extract_geo_names(sub_files).most_common()
    fu.dump_array("geo_freq_3_4_5_6_7.json", geo_freq)

from config.configure import getcfg
from classifying import fast_text_utils as ftu

import utils.array_utils as au
from utils.multiprocess_utils import CustomDaemonPool
import utils.tweet_keys as tk
import utils.pattern_utils as pu


# TODO one layer a daemon process(pool) ? First layer: filter, classification
value_t = ftu.value_t


def filter_twarr_text(twarr):
    for idx in range(len(twarr)-1, -1, -1):
        tw = twarr[idx]
        text_orgn = tw.get(tk.key_text).strip()
        text_norm = pu.text_normalization(text_orgn)
        if pu.is_empty_string(text_norm):
            twarr.pop(idx)
        tw.setdefault(tk.key_orgntext, text_orgn)
        tw.setdefault(tk.key_text, text_norm)
    return twarr


def filter_twarr_attr(twarr):
    flt_twarr = list()
    for idx, tw in enumerate(twarr):
        # TODO filter attributes for tweets
        flt_twarr.append(tw)
    return flt_twarr


def filter_twarr_fasttext(twarr, model, threshold):
    textarr = [tw.get(tk.key_text) for tw in twarr]
    pred_value_arr, score_arr = ftu.binary_predict(textarr, model, threshold)
    for idx in range(len(twarr) - 1, -1, -1):
        pred = pred_value_arr[idx]
        if pred != value_t:
            twarr.pop(idx)
    return twarr


""" actual function units """


def filter_and_classify(inq, outq):
    terror_model_file = getcfg().ft_terror_model_file
    ft_model = ftu.load_model(terror_model_file)
    while True:
        twarr = inq.get()
        # filter spam / ad / chat
        # len1 = len(twarr)
        twarr = filter_twarr_text(twarr)
        # len2 = len(twarr)
        twarr = filter_twarr_fasttext(twarr, ft_model, 0.4)
        # len3 = len(twarr)
        # print('{}->{}->{}'.format(len1, len2, len3))
        outq.put(twarr)


pool_size = 10
flt_cls_pool = CustomDaemonPool()
flt_cls_pool.start(filter_and_classify, pool_size)


def input_batches(tw_batches):
    flt_cls_pool.set_batch_input(tw_batches)


def output_batches():
    return flt_cls_pool.get_batch_output()

from preprocess.filter.yying_non_event_filter import EffectCheck
from classifying.terror.classifier_fasttext_add_feature import ClassifierAddFeature
from utils.multiprocess_utils import CustomDaemonPool
import utils.tweet_keys as tk
import utils.pattern_utils as pu


def filter_twarr_text(twarr):
    flt_twarr = list()
    for tw in twarr:
        text_orgn = tw.get(tk.key_text).strip()
        text_norm = pu.text_normalization(text_orgn)
        if pu.is_empty_string(text_norm) or not pu.has_azAZ(text_norm):
            continue
        tw.setdefault(tk.key_orgntext, text_orgn)
        tw.setdefault(tk.key_text, text_norm)
        flt_twarr.append(tw)
    return flt_twarr


def filter_twarr_attr(twarr):
    flt_twarr = list()
    for idx, tw in enumerate(twarr):
        # TODO filter attributes for tweets
        flt_twarr.append(tw)
    return flt_twarr


# def filter_twarr_fasttext(twarr, model, threshold):
#     textarr = [tw.get(tk.key_text) for tw in twarr]
#     pred_value_arr, score_arr = ftu.binary_predict(textarr, model, threshold)
#     for idx in range(len(twarr) - 1, -1, -1):
#         pred = pred_value_arr[idx]
#         if pred != value_t:
#             twarr.pop(idx)
#     return twarr


""" actual function units """


def filter_and_classify(inq, outq):
    ne_filter = EffectCheck()
    clf_filter = ClassifierAddFeature()
    while True:
        twarr = inq.get()
        len1 = len(twarr)
        twarr = filter_twarr_text(twarr)
        len2 = len(twarr)
        twarr = ne_filter.filter(twarr, 0.4)
        len3 = len(twarr)
        twarr = clf_filter.filter(twarr, 0.1)
        len4 = len(twarr)
        print('{}->{}'.format(len1, len4))
        outq.put(twarr)


pool_size = 15
flt_clf_pool = CustomDaemonPool()
flt_clf_pool.start(filter_and_classify, pool_size)


if __name__ == '__main__':
    EffectCheck()

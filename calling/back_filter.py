from preprocess.filter.yying_non_event_filter import EffectCheck
from classifying.terror.classifier_fasttext_add_feature import ClassifierAddFeature
from utils.multiprocess_utils import CustomDaemonPool
import utils.tweet_keys as tk
import utils.pattern_utils as pu


def filter_twarr_text(twarr):
    """ This function only suits for tweets that are not processed """
    flt_twarr = list()
    for tw in twarr:
        text_orgn = tw.get(tk.key_text, '').strip()
        text_norm = pu.text_normalization(text_orgn).strip()
        if pu.is_empty_string(text_norm) or not pu.has_enough_alpha(text_norm, 0.65):
            continue
        tw[tk.key_orgntext] = text_orgn
        tw[tk.key_text] = text_norm
        flt_twarr.append(tw)
    return flt_twarr


def filter_and_classify(inq, outq):
    ne_filter = EffectCheck()
    clf_filter = ClassifierAddFeature()
    while True:
        twarr = inq.get()
        len1 = len(twarr)
        twarr = filter_twarr_text(twarr)
        twarr = ne_filter.filter(twarr, 0.4)
        twarr = clf_filter.filter(twarr, 0.1)
        len4 = len(twarr)
        print('{}->{}'.format(len1, len4))
        outq.put(twarr)


pool_size = 15
flt_clf_pool = CustomDaemonPool()
flt_clf_pool.start(filter_and_classify, pool_size)


def set_batch_input(tw_batches):
    flt_clf_pool.set_batch_input(tw_batches)


def get_batch_output():
    return flt_clf_pool.get_batch_output()


def is_workload_num_over(workload_num):
    return flt_clf_pool.get_unread_batch_output_num() > workload_num

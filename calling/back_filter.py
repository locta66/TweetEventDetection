from preprocess.filter.yying_non_event_filter import EffectCheck
from classifying.terror.classifier_terror import ClassifierTerror
from utils.multiprocess_utils import CustomDaemonPool
import utils.tweet_keys as tk
import utils.pattern_utils as pu


flt_clf_pool = CustomDaemonPool()


def filter_twarr_text(twarr):
    """ This function only suits for tweets that are not processed """
    flt_twarr = list()
    for tw in twarr:
        # TODO text_orgn = tw.get(tk.key_text, '').strip()
        text_orgn = tw.get(tk.key_orgntext, tw.get(tk.key_text, None)).strip()
        if not text_orgn:
            continue
        text_norm = pu.text_normalization(text_orgn).strip()
        if pu.is_empty_string(text_norm) or not pu.has_enough_alpha(text_norm, 0.65):
            continue
        tw[tk.key_orgntext] = text_orgn
        tw[tk.key_text] = text_norm
        flt_twarr.append(tw)
    return flt_twarr


def filter_and_classify(inq, outq):
    idx, ne_threshold, clf_threshold = inq.get()
    outq.put(idx)
    ne_filter = EffectCheck()
    clf_filter = ClassifierTerror()
    while True:
        twarr = inq.get()
        # len1 = len(twarr)
        twarr = filter_twarr_text(twarr)
        flt_twarr = ne_filter.filter(twarr, ne_threshold)
        flt_twarr = clf_filter.filter(flt_twarr, clf_threshold)
        # len4 = len(twarr)
        # print('{}->{}'.format(len1, len4))
        outq.put(flt_twarr)


def start_pool(pool_size, ne_threshold, clf_threshold):
    flt_clf_pool.start(filter_and_classify, pool_size)
    flt_clf_pool.set_batch_input([(idx, ne_threshold, clf_threshold) for idx in range(pool_size)])
    flt_clf_pool.get_batch_output()


def input_twarr_batch(tw_batches):
    if tw_batches:
        flt_clf_pool.set_batch_input(tw_batches)


def get_batch_output():
    # Block until can get batch output
    return flt_clf_pool.get_batch_output()


def can_read_batch_output():
    return flt_clf_pool.can_read_batch_output()


def is_workload_num_over(workload_num):
    return flt_clf_pool.get_unread_batch_output_num() > workload_num

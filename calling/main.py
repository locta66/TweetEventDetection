import sys
import json
import time

from calling.back_filter import flt_clf_pool, pool_size
import calling.back_cluster as bclu
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.tweet_keys as tk
import utils.timer_utils as tmu


def read_lines():
    lines = sys.stdin.getlines()
    twarr = list()
    for line in lines:
        try:
            tw = json.loads(line, encoding='utf8')
            twarr.append(tw)
        except:
            continue
    return twarr


def twarr2filter(twarr):
    tw_batches = mu.split_multi_format(twarr, pool_size)
    flt_clf_pool.set_batch_input(tw_batches)


def filter2cluster():
    """ judging if get batch from pool may cause block should be done outside this function """
    filtered_batches = flt_clf_pool.get_batch_output()
    filtered_twarr = au.merge_array(filtered_batches)
    # bclu.input_twarr_batch(filtered_twarr)
    print("{} tweets filtered".format(len(filtered_twarr)))


# --del--
    preserve_key_info_of_twarr(filtered_twarr)
filtered_twarr_file = "filtered.json"
fi.remove_file(filtered_twarr_file)
def preserve_key_info_of_twarr(twarr):
    key_fields = {tk.key_id, tk.key_text, tk.key_orgntext, }
    filtered_twarr = [dict([(key, tw[key]) for key in key_fields]) for tw in twarr]
    fu.dump_array(filtered_twarr_file, filtered_twarr, overwrite=False)
# --del--


def ensure_filter_workload(workload_num=0):
    while flt_clf_pool.get_unread_batch_output_num() > workload_num:
        filter2cluster()


def wait_for_cluster():
    bclu.wait()


def main():
    sub_files = fi.listchildren("/home/nfs/cdong/tw/origin/", fi.TYPE_FILE, concat=True)[:800]
    _twarr_total_size = 0
    for idx, file in enumerate(sub_files):
        _twarr = fu.load_array(file)
        print("input twarr into filter: {}, length: {}".format(idx, len(_twarr)))
        _twarr_total_size += len(_twarr)
        twarr2filter(_twarr)
        ensure_filter_workload(3)
    
    ensure_filter_workload()
    print("total tw num", _twarr_total_size)
    print("wait for filter")
    # print("wait for cluster")
    # wait_for_cluster()


if __name__ == '__main__':
    tmu.check_time()
    main()
    tmu.check_time(print_func=lambda dt: print("time elapsed {}s".format(dt)))
    # 100 hours, 3914407 -> 93128 in 5695s(95m), threshold 0.1

import sys
import json
import time

from calling.back_filter import flt_clf_pool, pool_size
import calling.back_cluster as bclu
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
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
    tw_batches = au.array_partition(twarr, [1] * pool_size, random=False)
    flt_clf_pool.set_batch_input(tw_batches)
    # idx = 0
    # while not flt_clf_pool.can_get_batch_output():
    #     print('{:<6} {}'.format(idx, flt_clf_pool.get_ready_list()))
    #     time.sleep(1)
    #     idx += 1


def filter2cluster():
    """ Does not judge if get batch from pool may cause block,
        thus the judgement should be made outside the function """
    filtered_batches = flt_clf_pool.get_batch_output()
    filtered_twarr = au.merge_array(filtered_batches)
    bclu.input_twarr_batch(filtered_twarr)
    print('{} tweets filtered'.format(len(filtered_twarr)))

# --del--
    preserve_key_info_of_twarr(filtered_twarr)
filtered_twarr_file = 'filtered.json'
fi.remove_file(filtered_twarr_file)
def preserve_key_info_of_twarr(twarr):
    key_fields = {tk.key_text, tk.key_orgntext}
    filtered_twarr = [dict([(key, tw[key]) for key in key_fields]) for tw in twarr]
    fu.dump_array(filtered_twarr_file, filtered_twarr, overwrite=False)
# --del--


def try_filter2cluster():
    while flt_clf_pool.can_get_batch_output():
        filter2cluster()


def ensure_filter_workload(workload_num=0):
    while flt_clf_pool.get_unread_batch_output_num() > workload_num:
        filter2cluster()


def main():
    sub_files = fi.listchildren('/home/nfs/cdong/tw/origin/', fi.TYPE_FILE, concat=True)[:20]
    _twarr_total_size = 0
    for idx, file in enumerate(sub_files):
        _twarr = fu.load_array(file)
        print('input twarr batch: {}, length: {}'.format(idx, len(_twarr)))
        _twarr_total_size += len(_twarr)
        twarr2filter(_twarr)
        ensure_filter_workload(3)
    
    print('total tw num', _twarr_total_size)
    print('all tweets read, wait for filter')
    ensure_filter_workload()
    print('all tweets read, wait for cluster')
    bclu.wait()


if __name__ == '__main__':
    tmu.check_time()
    main()
    tmu.check_time(print_func=lambda dt: print('time elapsed {}s'.format(dt)))

import sys
import json

import calling.back_filter as bflt
import calling.back_cluster as bclu
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
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
    tw_batches = mu.split_multi_format(twarr, bflt.pool_size)
    bflt.set_batch_input(tw_batches)
    
# --del--
    fu.dump_array(history_twarr_len_file, [len(twarr)], overwrite=False)
output_base = './last_1000_without_gpe'
fi.mkdir(output_base)
filtered_twarr_file = fi.join(output_base, "filtered_twarr.json")
history_twarr_len_file = fi.join(output_base, "history_twarr_len.txt")
history_filter_len_file = fi.join(output_base, "history_filter_len.txt")
fi.remove_file(filtered_twarr_file)
fi.remove_file(history_filter_len_file)
fi.remove_file(history_twarr_len_file)
# --del--


def filter2cluster():
    """ judging if get batch from pool may cause block should be done outside this function """
    filtered_batches = bflt.get_batch_output()
    filtered_twarr = au.merge_array(filtered_batches)
    # bclu.input_twarr_batch(filtered_twarr)
    print("{} tweets filtered".format(len(filtered_twarr)))
    
# --del--
    fu.dump_array(filtered_twarr_file, filtered_twarr, overwrite=False)
    fu.dump_array(history_filter_len_file, [len(filtered_twarr)], overwrite=False)
# --del--


def ensure_filter_workload(workload_num=0):
    while bflt.is_workload_num_over(workload_num):
        filter2cluster()


def wait_for_cluster():
    bclu.wait()


def main():
    sub_files = fi.listchildren("/home/nfs/cdong/tw/origin/", fi.TYPE_FILE, concat=True)[-1000:]
    for idx, file in enumerate(sub_files):
        _twarr = fu.load_array(file)
        print("input twarr into filter: {}, length: {}".format(idx, len(_twarr)))
        twarr2filter(_twarr)
        ensure_filter_workload(5)
        if idx % 50 == 0:
            tmu.check_time('for idx, file in enumerate(sub_files)')
    ensure_filter_workload()
    # print("wait for cluster")
    # wait_for_cluster()


if __name__ == '__main__':
    tmu.check_time()
    main()
    tmu.check_time(print_func=lambda dt: print("time elapsed {}s".format(dt)))
    # 100  hours: 3914407  -> 93128  in 5695 s  (1.58 h),  threshold 0.1, thread 5
    # 1000 hours: 40110901 -> 988585 in 60769 s (16.88 h), threshold 0.1, thread 5
    # last 1000 hours: 37232413 -> 48193  in 7818 s (2.17 h), thread 15

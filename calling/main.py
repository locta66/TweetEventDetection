import calling.back_filter as bflt
import calling.back_cluster as bclu
import calling.back_extractor as bext
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.timer_utils as tmu
import utils.email_utils as emu


# filter & classify
flt_pool_size = 12
ne_threshold = 0.4
clf_threshold = 0.13
# cluster
hold_batch_num = 10
batch_size = 10
alpha = 50
beta = 0.005
# extractor
ext_pool_size = 8


def twarr2filter(twarr):
    """
    Read twarr into the filter
    :param twarr:
    :return:
    """
    tw_batches = mu.split_multi_format(twarr, flt_pool_size)
    bflt.input_twarr_batch(tw_batches)
# --del--
#     fu.dump_array(history_twarr_len_file, [len(twarr)], overwrite=False)
# output_base = './last_4000-ne_thres={}-clf_thres={}-hold_batch={}-batch_size={}-alpha={}-beta={}'.\
#     format(ne_threshold, clf_threshold, hold_batch_num, batch_size, alpha, beta)
# fi.mkdir(output_base, remove_previous=True)
# filtered_twarr_file = fi.join(output_base, "filtered_twarr.json")
# history_twarr_len_file = fi.join(output_base, "history_twarr_len.txt")
# history_filter_len_file = fi.join(output_base, "history_filter_len.txt")
# --del--


def ensure_filter_workload():
    while bflt.is_workload_num_over(0):
        filter2cluster()


def try_filter2cluster():
    while bflt.can_read_batch_output():
        filter2cluster()


def filter2cluster():
    """
    Read output from filter and transfer it to the cluster
    :return
    """
    filtered_batches = bflt.get_batch_output()
    filtered_twarr = au.merge_array(filtered_batches)
    print(len(filtered_twarr))
    bclu.input_twarr_batch(filtered_twarr)
    print('input to cluster over')
# --del--
#     fu.dump_array(filtered_twarr_file, filtered_twarr, overwrite=False)
#     fu.dump_array(history_filter_len_file, [len(filtered_twarr)], overwrite=False)
# --del--


def cluster2extractor():
    """
    Read output from cluster and transfer it to the extractor
    :return:
    """
    cluid_twarr_list = bclu.try_get_cluid_twarr_list()
    bext.input_cluid_twarr_list(cluid_twarr_list)


def main():
    bflt.start_pool(flt_pool_size, ne_threshold, clf_threshold)
    bclu.start_pool(hold_batch_num, batch_size, alpha, beta)
    # bext.start_pool(ext_pool_size)
    
    sub_files = fi.listchildren("/home/nfs/cdong/tw/origin/", fi.TYPE_FILE, concat=True)[-4000:]
    for _idx, _file in enumerate(sub_files):
        _twarr = fu.load_array(_file)
        print("1-- {} th twarr to filter, len: {}".format(_idx, len(_twarr)))
        twarr2filter(_twarr)
        # if _idx > 0 and (_idx + 1) % 1000 == 0:
        #     dt = tmu.check_time('if_idx>0and(_idx+1)%1000==0:', print_func=None)
        #     emu.send_email('notification', '{}/{} file, {}s from last 1000 file'.format(_idx+1, len(sub_files), dt))
        # if _idx % 50 == 0:
        #     tmu.check_time('_idx, _file', print_func=lambda dt: print("{} s from last 50".format(dt)))
        if _idx > 0 and _idx % 10 != 0:
            continue
        try_filter2cluster()
        
        # cluid_twarr_list = bclu.get_cluid_twarr_list()
        # print(len(cluid_twarr_list) if cluid_twarr_list else '--not ready')
        # if cluid_twarr_list:
        #     print(len(cluid_twarr_list))
    
    ensure_filter_workload()


if __name__ == '__main__':
    tmu.check_time()
    main()
    tmu.check_time(print_func=lambda dt: print("total time elapsed {}s".format(dt)))
    # 100  hours: 3914407  -> 93128  in 5695 s  (1.58 h),  threshold 0.1, thread 5
    # 1000 hours: 40110901 -> 988585 in 60769 s (16.88 h), threshold 0.1, thread 5
    # last 1000 hours: 37232413 -> 48193  in 7818 s (2.17 h), thread 15

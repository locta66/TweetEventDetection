from utils.multiprocess_utils import CustomDaemonPool
from extracting.cluster_infomation import ClusterInfoGetter, ClusterInfoCarrier

import utils.function_utils as fu


cluster_process_pool = CustomDaemonPool()


def per_cluid_twarr(inq, outq):
    pidx = inq.get()
    outq.put(pidx)
    cig = ClusterInfoGetter()
    while True:
        cluid, twarr = inq.get()
        extract_res = cig.input_cluid_twarr(cluid, twarr)
        cic = ClusterInfoCarrier(*extract_res)
        outq.put(cic)


def start_pool(pool_size):
    cluster_process_pool.start(per_cluid_twarr, pool_size)
    cluster_process_pool.set_batch_input([i for i in range(pool_size)])
    cluster_process_pool.get_batch_output()


def input_cluid_twarr_list(cluid_twarr_list):
    if cluid_twarr_list:
        cluster_process_pool.set_batch_input(cluid_twarr_list)


def get_batch_output():
    # Blocks until can get batch output
    return cluster_process_pool.get_batch_output()


def can_read_batch_output():
    return cluster_process_pool.can_read_batch_output()


if __name__ == '__main__':
    import time
    import utils.timer_utils as tmu
    import utils.file_iterator as fi
    # fi.mkdir('/home/nfs/cdong/tw/src/calling/ext_tmp', remove_previous=True)
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive'
    pos_files = fi.listchildren(base, concat=True)[:14]
    
    start_pool(8)
    input_cluid_twarr_list([(i, fu.load_array(file)) for i, file in enumerate(pos_files)])
    
    tmu.check_time()
    while not cluster_process_pool.can_read_batch_output():
        print('sleeping')
        time.sleep(5)
        continue
    # get_extract_batch_output()
    print('trying to get')
    batch_output = cluster_process_pool.get_batch_output()
    print('get batch outputs')
    print('read output', [(c.cluid, c.od['summary']['keywords']) for c in batch_output])
    tmu.check_time()


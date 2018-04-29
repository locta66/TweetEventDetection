from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic
from utils.multiprocess_utils import CustomDaemonProcess
import multiprocessing as mp


class ClusterDaemonProcess(CustomDaemonProcess):
    def __init__(self, pidx=0):
        CustomDaemonProcess.__init__(self, pidx)
        self.outq2 = mp.Queue()
    
    def start(self, func):
        self.process = mp.Process(target=func, args=(self.inq, self.outq, self.outq2))
        self.process.daemon = True
        self.process.start()


END_PROCESS = -1
INPUT_TWARR = 0
SET_PARAMS = 1
cluster_daemon = ClusterDaemonProcess()


def clustering(inq, outq, outq_list):
    clusterer = BackCluster()
    while True:
        command = inq.get()
        if command == INPUT_TWARR:
            tw_batch = inq.get()
            clusterer.input_twarr(tw_batch, outq_list)
        elif command == SET_PARAMS:
            params = inq.get()
            clusterer.set_parameters(*params)
        elif command == END_PROCESS:
            outq.put('ending')
            return
        else:
            print('no such command')


def send_simple_message(command, arg, receive):
    cluster_daemon.set_input(command)
    if arg:
        cluster_daemon.set_input(arg)
    if receive:
        return cluster_daemon.get_output()


def start_pool(hold_batch_num, batch_size, alpha, beta):
    cluster_daemon.start(clustering)
    send_simple_message(SET_PARAMS, arg=(hold_batch_num, batch_size, alpha, beta), receive=False)


def wait():
    send_simple_message(END_PROCESS, arg=None, receive=True)
    cluster_daemon.end()


def input_twarr_batch(tw_batch):
    if tw_batch:
        send_simple_message(INPUT_TWARR, arg=tw_batch, receive=False)


def wait_until_get_cluid_twarr_list():
    return cluster_daemon.outq2.get()


def try_get_cluid_twarr_list():
    cluid_twarr_list = None
    while cluster_daemon.outq2.qsize() > 0:
        cluid_twarr_list = wait_until_get_cluid_twarr_list()
    return cluid_twarr_list


class BackCluster:
    def __init__(self):
        self.read_batch_num = self.hist_len = 0
        self.hold_batch_num = self.batch_size = None
        self.gsdpmm = GSDPMMStreamIFDDynamic()
        self.tweet_pool = list()
    
    def set_parameters(self, hold_batch_num, batch_size, alpha, beta):
        self.hold_batch_num, self.batch_size = hold_batch_num, batch_size
        self.gsdpmm.set_hyperparams(alpha, beta)
    
    def input_twarr(self, twarr, out_channel):
        twarr = self.gsdpmm.filter_dup_id(twarr)
        self.tweet_pool.extend(twarr)
        n = len(twarr)
        self.hist_len += n
        print('  -> new {} tw, current pool: {}, hist len: {}'.format(n, len(self.tweet_pool), self.hist_len))
        while len(self.tweet_pool) >= self.batch_size:
            self.input_batch(self.tweet_pool[:self.batch_size])
            self.tweet_pool = self.tweet_pool[self.batch_size:]
            self.output(out_channel)
    
    def input_batch(self, tw_batch):
        self.read_batch_num += 1
        print('  - read batch num: {}'.format(self.read_batch_num))
        params = [
            dict(tw_batch=tw_batch, action=GSDPMMStreamIFDDynamic.ACT_STORE, iter_num=None),
            dict(tw_batch=tw_batch, action=GSDPMMStreamIFDDynamic.ACT_FULL, iter_num=25),
            dict(tw_batch=tw_batch, action=GSDPMMStreamIFDDynamic.ACT_SAMPLE, iter_num=3)
        ]
        hbn, rbn, = self.hold_batch_num, self.read_batch_num
        param = params[0 if rbn < hbn else 1 if rbn == hbn else 2]
        self.gsdpmm.input_batch(**param)
    
    def output(self, out_channel):
        hbn, rbn = self.hold_batch_num, self.read_batch_num
        interval, twnum_thres = 5, 4
        if not (rbn >= hbn and (rbn - hbn) % interval == 0):
            return
        print('new list generated, hbn={}, rbn={}'.format(hbn, rbn))
        cluid_twarr_list = self.gsdpmm.get_cluid_twarr_list(twnum_thres)
        if cluid_twarr_list:
            out_channel.put(cluid_twarr_list)


if __name__ == '__main__':
    import utils.tweet_keys as tk
    import utils.array_utils as au
    import utils.pattern_utils as pu
    import utils.timer_utils as tmu
    import calling.back_extractor as bext
    import utils.file_iterator as fi
    import utils.function_utils as fu
    fi.mkdir('/home/nfs/cdong/tw/src/calling/tmp', remove_previous=True)
    
    tmu.check_time()
    _hold_batch_num = 100
    _batch_size = 100
    _alpha, _beta = 30, 0.01
    # _alpha, _beta = 50, 0.005
    _file = "./filtered_twarr.json"
    _twarr = fu.load_array(_file)[:10200]
    start_pool(_hold_batch_num, _batch_size, _alpha, _beta)
    input_twarr_batch(_twarr)
    
    print('---> waiting for _cluid_cluster_list')
    while True:
        _cluid_cluster_list = cluster_daemon.outq2.get()
        print('     - some thing returned, type :{}'.format(type(_cluid_cluster_list)))
        if _cluid_cluster_list is not None:
            break
    print('---> get _cluid_cluster_list, len:{}'.format(len(_cluid_cluster_list)))
    
    _ext_pool_size = 10
    bext.start_pool(_ext_pool_size)
    bext.input_cluid_twarr_list(_cluid_cluster_list)
    print('waiting for cic outputs')
    _cic_list = bext.get_batch_output()
    print('get cic outputs, type:{}'.format(type(_cic_list)))
    for cic in _cic_list:
        twnum = len(cic.twarr)
        _geo_list = [geo['address'] for geo in cic.od['geo_infer'] if geo['quality'] == 'locality']
        print('cluid:{}, twarr len:{}'.format(cic.cluid, twnum))
        print(cic.od['summary']['keywords'])
        print(_geo_list)
        print('\n')
        
        if len(_geo_list) == 0:
            _top_geo = 'NOGPE'
        else:
            _top_geo = '`'.join(_geo_list)
        _out_file = '/home/nfs/cdong/tw/src/calling/tmp/id{}_tw{}_{}.txt'.format(cic.cluid, twnum, _top_geo)
        _txtarr = [tw[tk.key_text] for tw in cic.twarr]
        _idx_g, _txt_g = au.group_similar_items(_txtarr, score_thres=0.3, process_num=20)
        _txt_g = [sorted(g, key=lambda t: len(t), reverse=True) for g in _txt_g]
        _txtarr = au.merge_array(_txt_g)
        fu.write_lines(_out_file, _txtarr)
    
    tmu.check_time()

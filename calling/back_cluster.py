from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic
from utils.multiprocess_utils import CustomDaemonProcess
import utils.tweet_utils as tu


CMD_INPUT_TWARR = 0
CMD_WRITE_CLUSTER = 1
CMD_END_PROCESS = -1

hold_batch_num = 5
batch_size = 100


def cluster_and_extract(inq, outq):
    g = GSDPMMStreamIFDDynamic()
    hyperparams = inq.get()
    g.set_hyperparams(*hyperparams)
    twarr_pool = list()
    
    gsdpmm_batch_num = 0
    histroy_len = 0
    while True:
        command = inq.get()
        if command == CMD_INPUT_TWARR:
            twarr = inq.get()
            histroy_len += len(twarr)
            tu.twarr_nlp(twarr)
            twarr_pool.extend(twarr)
            print('{} received new {} tweets, current pool size: {}, total input len: {}'.
                  format(gsdpmm_batch_num, len(twarr), len(twarr_pool), histroy_len))
            while len(twarr_pool) >= batch_size:
                gsdpmm_batch_num += 1
                print('gsdpmm batch num: {}'.format(gsdpmm_batch_num))
                g.input_batch(twarr_pool[:batch_size])
                twarr_pool = twarr_pool[batch_size:]
        elif command == CMD_WRITE_CLUSTER:
            file = inq.get()
            # TODO
            print(file)
        elif command == CMD_END_PROCESS:
            outq.put('ready to end')
            return


cluster_daemon = CustomDaemonProcess()
cluster_daemon.start(cluster_and_extract)
cluster_daemon.set_input((hold_batch_num, 30, 0.01))


def input_twarr_batch(twarr):
    cluster_daemon.set_input(CMD_INPUT_TWARR)
    cluster_daemon.set_input(twarr)


def write_clusters_into_file(file):
    cluster_daemon.set_input(CMD_WRITE_CLUSTER)
    cluster_daemon.set_input(file)


def wait():
    cluster_daemon.set_input(CMD_END_PROCESS)
    cluster_daemon.get_output()
    cluster_daemon.end()

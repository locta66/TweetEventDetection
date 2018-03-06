from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic
from utils.multiprocess_utils import CustomDaemonProcess
import utils.tweet_utils as tu


# TODO second layer: clustering, feature extracting


def cluster_and_extract(inq, outq):
    g = GSDPMMStreamIFDDynamic()
    hyperparams = inq.get()
    g.set_hyperparams(*hyperparams)
    while True:
        twarr = inq.get()
        tu.twarr_nlp(twarr)
        g.input_batch(twarr)


cluster_daemon = CustomDaemonProcess(cluster_and_extract)
cluster_daemon.start()
hold_batch_num = 5
cluster_daemon.set_input((hold_batch_num, 30, 0.01))


def input_batch(tw_batch):
    cluster_daemon.set_input(tw_batch)

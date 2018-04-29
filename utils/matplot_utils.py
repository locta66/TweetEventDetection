import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import utils.function_utils as fu


def figure(X, Y, fig_name):
    # plt.figure(figsize=(13, 7))
    plt.plot(X, Y, color="blue", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("roc curve")
    plt.legend(loc='lower right')
    plt.savefig(fig_name, format='png')


if __name__ == '__main__':
    import utils.array_utils as au
    labelarr, probarr = fu.load_array("/home/nfs/cdong/tw/src/preprocess/filter/prb_lbl_arr.txt")
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labelarr, probarr)
    figure(fpr, tpr, "/home/nfs/cdong/tw/src/preprocess/filter/roc_curve.png")
    au.precision_recall_threshold(labelarr, probarr, file="/home/nfs/cdong/tw/src/preprocess/filter/performance.csv",
                                  thres_range=[i / 100 for i in range(1, 10)] + [i / 20 for i in range(2, 20)])

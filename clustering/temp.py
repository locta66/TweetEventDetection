import sklearn
import re
from sklearn import svm, metrics
import utils.tweet_utils as tu
import utils.function_utils as fu
import utils.file_iterator as fi
import utils.tweet_keys as tk
import utils.multiprocess_utils as mu
import utils.ark_service_proxy as ark
import utils.array_utils as au
import re


def multi(file):
    # ent_tags = {'FAC', 'GPE', 'LOC', 'ORG', 'NORP'}
    word_type = list()
    twarr = fu.load_array(file)
    twarr = tu.twarr_nlp(twarr)
    for tw in twarr:
        doc = tw[tk.key_spacy]
        for token in doc:
            word_type.append([token.text, token.ent_type_, token.tag_])
    return word_type


def func2():
    file = '/home/nfs/cdong/tw/src/clustering/data/events.txt'
    twarr_blocks = fu.load_array(file)
    for tw in twarr_blocks[19]:
        print(tw[tk.key_text])
    # print()
    # print()
    # for tw in twarr_blocks[56]:
    #     print(tw[tk.key_text])
    # print()
    # print()
    # for tw in twarr_blocks[70]:
    #     print(tw[tk.key_text])
    # print()
    # print()


def func3():
    str_arr = fu.load_array('sim_info.txt')
    feature = list()
    labels = list()
    for string in str_arr:
        num_arr = [float(s) for s in re.findall('\d\.\d+|\d+', string)]
        # if num_arr[4] < 0.5:
        #     continue
        feature.append([num_arr[1], num_arr[3], num_arr[4]])
        labels.append(1 if num_arr[0] == num_arr[2] else 0)
        print(num_arr, feature[-1], labels[-1])
    
    split_idx = int(len(feature) * 0.3)
    trainX, testX = feature[split_idx:], feature[:split_idx]
    trainY, testY = labels[split_idx:], labels[:split_idx]
    
    clf = svm.SVC()
    # clf.fit(feature, labels)
    # predY = clf.predict(feature)
    # auc = sklearn.metrics.roc_auc_score(labels, predY)
    
    clf.fit(trainX, trainY)
    predY = clf.predict(testX)
    auc = sklearn.metrics.roc_auc_score(testY, predY)
    print(auc)
    for idx in range(len(predY)):
        print(predY[idx], testY[idx])
    
    precision, recall, thresholds = metrics.precision_recall_curve(testY, predY)
    
    last_idx = 0
    for ref in [i / 10 for i in range(3, 8)]:
        for idx in range(last_idx, len(thresholds)):
            if thresholds[idx] >= ref:
                print('threshold', round(thresholds[idx], 2), '\tprecision', round(precision[idx], 5),
                      '\trecall', round(recall[idx], 5))
                last_idx = idx
                break
    
    # print('--pred--')
    # wrong = 0
    # for idx in range(len(predY)):
    #     p = predY[idx]
    #     l = labels[idx]
    #     if not p == l:
    #         print(feature[idx], p, l)
    #         wrong += 1
    # print(len(feature), wrong)
    # print("pred {}, true {}".format(p, l))
    # print(clf.predict(feature))


def identify_korea():
    file = '/home/nfs/cdong/tw/seeding/NorthKorea/korea.json'
    twarr_blocks = fu.load_array(file)
    twarr = au.merge_array(twarr_blocks)
    for tw in twarr:
        text = tw[tk.key_text]
        if not re.search('korea', text, flags=re.I):
            print(text)


if __name__ == '__main__':
    func2()
    exit()
    # from multiprocessing import Process, Value, Array
    # twarr = fu.load_array('/home/nfs/yangl/event_detection/testdata/event2012/sorted_relevant.json')[:10]
    
    # file = '/home/nfs/cdong/tw/src/clustering/data/events2012.txt'
    # file = '/home/nfs/cdong/tw/src/clustering/data/events.txt'
    # file = '/home/nfs/cdong/tw/src/clustering/data/events2016.txt'
    # print(len(au.merge_list(fu.load_array(file))))
    # file = '/home/nfs/cdong/tw/src/clustering/data/falseevents.txt'
    # print(len(fu.load_array(file)))
    
    # process_korea()
    identify_korea()
    exit()
    
    # twarr = fu.load_array('/home/nfs/yangl/event_detection/testdata/event2012/sorted_relevant.json')
    # twarr = fu.load_array('/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_counter.sum')
    # print(len(twarr))
    # func2()
    
    from collections import Counter
    from clustering.main2clusterer import create_korea_batches_through_time
    
    tw_batches, lb_batches = create_korea_batches_through_time(1000)
    for lbb in lb_batches:
        c = Counter(lbb)
        print([(eid, c[eid]) for eid in sorted(c.keys())])

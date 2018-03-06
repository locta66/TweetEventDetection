import utils.file_iterator as fi
import utils.spacy_utils as su
import utils.array_utils as au
import utils.function_utils as fu
import utils.pattern_utils as pu
import utils.tweet_keys as tk
import utils.multiprocess_utils as mu
import utils.timer_utils as tmu

import classifying.fast_text_make as ftm
import classifying.fast_text_utils as ftu

from classifying.terror.classifier import predict_proba

import numpy as np


def docarr2matrix(docarr):
    vecarr = list()
    for doc in docarr:
        docvec = np.concatenate(su.get_doc_pos_vectors(doc))
        vecarr.append(docvec)
    return np.array(vecarr)


def twarr2docarr(twarr):
    textarr = list()
    for tw in twarr:
        text = tw.get(tk.key_text).strip()
        if not pu.is_empty_string(text):
            textarr.append(text)
    return su.textarr_nlp(textarr)


def twarr2textarr(twarr):
    textarr = list()
    for tw in twarr:
        text = tw.get(tk.key_text).strip()
        if pu.is_empty_string(text):
            continue
        textarr.append(text)
    return textarr


def textarr_normalization(textarr):
    norm_textarr = list()
    for text in textarr:
        text = pu.text_normalization(text)
        if pu.is_empty_string(text):
            continue
        norm_textarr.append(text)
    return norm_textarr
    

def twarr2matrix(twarr):
    return docarr2matrix(twarr2docarr(twarr))


""" -------- for sklearn -------- """

pos_event_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/{}'
neg_files = ['/home/nfs/yying/data/crawlTwitter/Crawler1/test.json',
             '/home/nfs/yying/data/crawlTwitter/Crawler2/crawl2.json',
             '/home/nfs/yying/data/crawlTwitter/Crawler3/crawl3.json',
             '/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_counter.sum']


def make_positive_matrix():
    base_pattern = pos_event_pattern
    pos_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/pos_{:0>2}_mtx'
    pos_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE)
    for file_idx, file in enumerate(pos_files):
        matrix = twarr2matrix(fu.load_array(base_pattern.format(file)))
        print(file, matrix.shape)
        pos_file_name = pos_pattern.format(file_idx)
        np.save(pos_file_name, matrix)


def make_negative_matrix():
    neg_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/neg_{:0>2}_mtx'
    for file_idx, file in enumerate(neg_files):
        matrix = twarr2matrix(fu.load_array(file))
        print(file, matrix.shape)
        pos_file_name = neg_pattern.format(file_idx)
        np.save(pos_file_name, matrix)


def get_false_positive():
    """ make matrices """
    twarr = list()
    for neg_file in neg_files:
        neg_twarr = fu.load_array(neg_file)
        print(len(neg_twarr))
        if len(neg_twarr) > 50000:
            neg_twarr = au.random_array_items(neg_twarr, 80000)
        twarr.extend(neg_twarr)
    print('neg twarr load over')
    tmu.check_time()
    matrix = twarr2matrix(twarr)
    tmu.check_time()
    print('spacy over', len(twarr))
    
    """ make predictions """
    preds = predict_proba(matrix)
    assert len(twarr) == len(preds)
    false_pos_idx = list()
    for idx in range(len(twarr)):
        pred = preds[idx]
        if pred > 0.4:
            false_pos_idx.append(idx)
    false_pos_twarr = [twarr[idx] for idx in false_pos_idx]
    tw_attr_set = {tk.key_text, tk.key_id, tk.key_created_at, tk.key_orgntext,
                   tk.key_in_reply_to_status_id, tk.key_user}
    usr_attr_set = {"screen_name", "friends_count", "statuses_count", "description", "id"}
    for tw in false_pos_twarr:
        for tw_k in list(tw.keys()):
            if tw_k not in tw_attr_set:
                tw.pop(tw_k)
            user = tw[tk.key_user]
            for usr_k in list(user.keys()):
                if usr_k not in usr_attr_set:
                    user.pop(usr_k)
        print(tw[tk.key_text])
    print('fp rate: {}/{}'.format(len(false_pos_idx), len(twarr)))
    fu.dump_array('/home/nfs/cdong/tw/src/clustering/data/false_pos_events.txt', false_pos_twarr)


""" -------- for fasttext -------- """
label_t, label_f = ftu.label_t, ftu.label_f


def make_negative_event():
    neg_twarr_blocks = [fu.load_array(file) for file in neg_files]
    print([len(a) for a in neg_twarr_blocks], sum([len(a) for a in neg_twarr_blocks]))
    neg_textarr_blocks = [twarr2textarr(twarr) for twarr in neg_twarr_blocks]
    neg_textarr_blocks = [textarr_normalization(textarr) for textarr in neg_textarr_blocks]
    neg_label_blocks = [ftm.prefix_textarr_with_label(label_f, textarr) for textarr in neg_textarr_blocks]
    
    blocks = list()
    for label_textarr in neg_label_blocks:
        partition = au.array_partition(label_textarr, [1] * 5, random=False)
        blocks.extend(partition)
    print([len(a) for a in blocks], sum([len(a) for a in blocks]))
    
    neg_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/neg_{:0>2}.txt'
    for idx, label_textarr in enumerate(blocks):
        ftm.dump_textarr(neg_pattern.format(idx), label_textarr)


def make_positive_event():
    pos_files = fi.listchildren(pos_event_pattern.format(''), fi.TYPE_FILE)
    pos_twarr_blocks = [fu.load_array(pos_event_pattern.format(file)) for file in pos_files]
    pos_textarr_blocks = [twarr2textarr(twarr) for twarr in pos_twarr_blocks]
    pos_label_blocks = [ftm.prefix_textarr_with_label(label_t, textarr) for textarr in pos_textarr_blocks]
    
    blocks = pos_label_blocks
    print([len(a) for a in blocks], sum([len(a) for a in blocks]))
    pos_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/pos_{:0>2}.txt'
    for idx, label_textarr in enumerate(blocks):
        ftm.dump_textarr(pos_pattern.format(idx), label_textarr)


def make_train_test():
    base_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/{}'
    p_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE, pattern='pos')
    p_files = [base_pattern.format(file) for file in p_files]
    n_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE, pattern='neg')
    n_files = [base_pattern.format(file) for file in n_files]
    print(len(p_files), len(n_files))
    train_file = base_pattern.format('train')
    test_file = base_pattern.format('test')
    fi.concat_files(p_files[:47] * 10 + n_files[:10] + n_files[12:15], train_file)
    fi.concat_files(p_files[47:] + n_files[10:12] + n_files[15:], test_file)


def test_train_pos_neg_portion():
    def portion_of_file(file):
        p_cnt = n_cnt = 0
        with open(file, 'r') as fp:
            for idx, line in enumerate(fp.readlines()):
                label = line.split(' ', 1)[0]
                if label == label_t:
                    p_cnt += 1
                elif label == label_f:
                    n_cnt += 1
                else:
                    print(idx)
        return p_cnt, n_cnt
    train_p, train_n = portion_of_file('/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/train')
    print('train {}/{}={}'.format(train_p, train_n, train_p/train_n))
    test_p, test_n = portion_of_file('/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/test')
    print('test {}/{}={}'.format(test_p, test_n, test_p/test_n))


def f():
    base_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/{}'
    sub_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE, 'txt$')
    print(sub_files)
    lb_map = {'__label__t': label_t, '__label__f': label_f}
    for file in sub_files:
        with open(base_pattern.format(file)) as fp:
            lb_textarr = fp.readlines()
        textarr = list()
        for lb_text in lb_textarr:
            sp = lb_text.split(' ')
            sp[0] = lb_map[sp[0]]
            textarr.append(' '.join(sp))
        with open(base_pattern.format(file), 'w') as fp:
            fp.writelines(textarr)


if __name__ == '__main__':
    """ fasttext """
    make_negative_event()
    make_positive_event()
    make_train_test()
    # test_train_pos_neg_portion()
    # f()
    exit()

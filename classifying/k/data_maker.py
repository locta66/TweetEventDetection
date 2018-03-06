import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.tweet_keys as tk
import preprocess.tweet_filter as tflt
import classifying.fast_text_utils as ftu


label_t, label_f = ftu.label_t, ftu.label_f


def textarr2file(textarr, file, label):
    lines = list()
    for text in textarr:
        text = text.strip()
        if pu.is_empty_string(text):
            continue
        lines.append('{} {}\n'.format(label, text))
    with open(file, 'w') as fp:
        fp.writelines(lines)


def create_negative_korea():
    base_path = '/home/nfs/cdong/tw/origin/'
    out_template = '/home/nfs/cdong/tw/seeding/NorthKorea/data/neg_k_{}.txt'
    p_num = 10
    sub_files = au.random_array_items(fi.listchildren(base_path, fi.TYPE_FILE), 160)
    print(sub_files)
    file_block = mu.split_multi_format([base_path + sub_file for sub_file in sub_files], p_num)
    out_block = [out_template.format(idx) for idx in range(p_num)]
    k_cnt_list = mu.multi_process(create_negative_korea_single, [(file_block[idx], out_block[idx]) for idx in range(p_num)])
    # daepool = mu.ProxyDaemonPool()
    # daepool.start(create_non_korea, process_num)
    # daepool.set_batch_input([(file_list_block[idx], out_file_block[idx]) for idx in range(process_num)])
    # nk_count_list = daepool.get_batch_output()
    print(len(k_cnt_list), sum(k_cnt_list))


def create_negative_korea_single(file_list, out_file):
    textarr = list()
    is_k_count = 0
    for file in file_list:
        twarr = fu.load_array(file)
        for tw in twarr:
            text = tw.get(tk.key_text)
            if pu.search_pattern('korea', text) is not None or pu.search_pattern('north korea', text) is not None:
                is_k_count += 1
                continue
            textarr.append(text)
    textarr2file(textarr, out_file, label_f)
    return is_k_count


def process_korea_origin():
    twmarr = fu.load_array('/home/nfs/cdong/tw/seeding/NorthKorea/kr.json')
    e_info = dict()
    for idx, tw_meta in enumerate(twmarr):
        tw = tw_meta['tweet']
        user = tw['user']
        tw[tk.key_orgntext] = tw.pop('raw_text')
        tw[tk.key_text] = pu.tw_rule_patterns.apply_patterns(tw.pop('standard_text'))
        tw[tk.key_created_at] = tw[tk.key_created_at]['$date']
        tw['event_id'] = tw_meta['event_id']
        user.pop('userbadges')
        user.pop('avatar_src')
        user.pop('data_name')
        tw.pop('media')
        tw.pop('conversation_id')
        label = tw['event_id'] - 24
        description = tw_meta['q']
        if label not in e_info:
            e_info[label] = {}
            e_info[label]['desc'] = description
            e_info[label]['twarr'] = [tw]
        else:
            e_info[label]['twarr'].append(tw)
    labels = sorted(e_info.keys())
    fu.dump_array('/home/nfs/cdong/tw/seeding/NorthKorea/korea.json', [e_info[lb]['twarr'] for lb in labels])
    fu.dump_array('/home/nfs/cdong/tw/seeding/NorthKorea/info.json', [(lb, e_info[lb]['desc']) for lb in labels])


def create_positive_korea():
    twarr_blocks = fu.load_array('/home/nfs/cdong/tw/seeding/NorthKorea/korea.json')
    for idx, twarr in enumerate(twarr_blocks):
        tflt.filter_twarr_dup_id(twarr)
        out_file = '/home/nfs/cdong/tw/seeding/NorthKorea/data/pos_k_{}.txt'.format(idx)
        textarr2file([tw.get(tk.key_text).strip() for tw in twarr], out_file, label_t)


def create_event():
    natural = '/home/nfs/cdong/tw/seeding/NaturalDisaster/queried/NaturalDisaster.sum'
    terror_template = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/{}'
    terror_files = fi.listchildren(terror_template.format(''), fi.TYPE_FILE)
    twarr = au.merge_array([fu.load_array(terror_template.format(sub)) for sub in terror_files])
    twarr.extend(fu.load_array(natural))
    textarr = [tw.get(tk.key_text) for tw in twarr]
    sep = int(0.5 * len(textarr))
    textarr2file(textarr[:sep], '/home/nfs/cdong/tw/seeding/NorthKorea/data/event_1.txt', label_f)
    textarr2file(textarr[sep:], '/home/nfs/cdong/tw/seeding/NorthKorea/data/event_2.txt', label_f)


def concat_train_test():
    base_template = '/home/nfs/cdong/tw/seeding/NorthKorea/data/{}'
    neg_files = fi.listchildren(base_template.format(''), fi.TYPE_FILE, '^neg')
    pos_files = fi.listchildren(base_template.format(''), fi.TYPE_FILE, '^pos')
    evn_files = fi.listchildren(base_template.format(''), fi.TYPE_FILE, '^event')
    train_file_list = [base_template.format(file) for file in pos_files[:2] + evn_files[:1] + neg_files[:7]]
    test_file_list = [base_template.format(file) for file in pos_files[2:] + evn_files[1:] + neg_files[7:]]
    fi.concat_files(train_file_list, base_template.format('train'))
    fi.concat_files(test_file_list, base_template.format('test'))


def test_train_pos_neg_portion():
    def portion_of_file(file):
        p_cnt = n_cnt = 0
        with open(file, 'r') as fp:
            for line in fp.readlines():
                label = line.split(' ', 1)[0]
                if label == label_t:
                    p_cnt += 1
                elif label == label_f:
                    n_cnt += 1
                else:
                    print('label not in consider')
        return p_cnt, n_cnt
    train_file = '/home/nfs/cdong/tw/seeding/NorthKorea/data/train'
    test_file = '/home/nfs/cdong/tw/seeding/NorthKorea/data/test'
    train_p, train_n = portion_of_file(train_file)
    test_p, test_n = portion_of_file(test_file)
    print('train {}/{}={}'.format(train_p, train_n, train_p/train_n))
    print('test {}/{}={}'.format(test_p, test_n, test_p/test_n))


if __name__ == '__main__':
    create_negative_korea()
    create_positive_korea()
    create_event()
    concat_train_test()
    # test_train_pos_neg_portion()

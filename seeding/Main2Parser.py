import ArrayUtils
import FileIterator
import TweetKeys
from EventFeatureExtractor import EventFeatureExtractor
from WordFreqCounter import WordFreqCounter

import Levenshtein


def exec_query(data_path, parser):
    data_path = FileIterator.append_slash_if_necessary(data_path)
    FileIterator.iterate_file_tree(data_path, query_tw_files_in_path_multi, parser=parser)
    print('Queried', len(parser.added_twarr), 'tweets,', end=' ')
    remove_similar_tws(parser.added_twarr)
    print(len(parser.added_twarr), 'accepted.\n', parser.get_query_result_file_name(), 'written.\n')
    FileIterator.dump_array(parser.get_query_result_file_name(), parser.added_twarr)
    # for tw in parser.added_twarr:
    #     print(tw[TweetKeys.key_cleantext], '\n')


def query_tw_files_in_path_multi(json_path, *args, **kwargs):
    parser = kwargs['parser']
    subfiles = FileIterator.listchildren(json_path, children_type='file')
    file_list = [(json_path + file_name) for file_name in subfiles if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    if not file_list:
        return
    print(file_list[0], '\n', len(file_list), 'files until', '\n', file_list[-1])
    file_list = FileIterator.split_into_multi_format(file_list, process_num=16)
    added_twarr_block = FileIterator.multi_process(query_tw_file_multi,
        [(file_list_slice, parser.__class__, parser.query_list, parser.theme,
          parser.description) for file_list_slice in file_list])
    parser.added_twarr = FileIterator.merge_list(added_twarr_block)


def query_tw_file_multi(file_list, parser_class, query_list, theme, description):
    # Query tweets that satisfies condition of parser from a given file set
    parser = parser_class(query_list, theme, description)
    for file in file_list:
        parser.read_tweet_from_json_file(file)
    return parser.added_twarr


def remove_similar_tws(twarr):
    for i in range(len(twarr)-1, -1, -1):
        for j in range(len(twarr)-1, i, -1):
            istr = twarr[i][TweetKeys.key_cleantext].lower()
            jstr = twarr[j][TweetKeys.key_cleantext].lower()
            dist = Levenshtein.distance(istr, jstr) + 1
            if max(len(istr), len(jstr)) / dist >= 5:
                # print('[', twarr[j][TweetKeys.key_cleantext], ']')
                del twarr[j]
    return twarr


# def query_tw_files_in_path(json_path, *args, **kwargs):
#     parser = kwargs['parser']
#     subfiles = FileIterator.listchildren(json_path, children_type='file')
#     for subfile in subfiles:
#         if not subfile.endswith('.sum'):
#             continue
#         json_file = json_path + subfile
#         parser.read_tweet_from_json_file(json_file)


def exec_totag(parser):
    tw_file = parser.get_query_result_file_name()
    reset_tag_of_tw_file(tw_file)
    to_tag_dict = dict()
    for tw in FileIterator.load_array(tw_file):
        to_tag_dict[tw['id']] = tw[TweetKeys.key_origintext]
    FileIterator.dump_array(parser.get_to_tag_file_name(),
                            [parser.theme, parser.description, parser.get_theme_path(), to_tag_dict], sort_keys=True)


def reset_tag_of_tw_file(tw_file):
    tw_arr = FileIterator.load_array(tw_file)
    for tw in tw_arr:
        tw[TweetKeys.key_ptagtime] = 0
        tw[TweetKeys.key_ntagtime] = 0
        tw[TweetKeys.key_tagtimes] = 0
    FileIterator.dump_array(tw_file, tw_arr)


def exec_ner(parser):
    efe = EventFeatureExtractor()
    efe.start_ner_service()
    efe.perform_ner_on_tw_file(parser.get_query_result_file_name())
    efe.end_ner_service()


def exec_untag(parser):
    update_tw_file_from_tag_in_path(parser.get_query_result_file_name(), parser.get_theme_path())


def update_tw_file_from_tag_in_path(tw_file, tagged_file_path, tagged_postfix='.tag',
                                    output_to_another_file=False, another_file='./deafult.sum'):
    tw_arr = FileIterator.load_array(tw_file)
    tw_arr_dict = dict([(tw[TweetKeys.key_id], tw) for tw in tw_arr])
    for file in FileIterator.listchildren(tagged_file_path, children_type='file'):
        if not file.endswith(tagged_postfix):
            continue
        update_tw_arr_dict_from_tagged_file(tw_arr_dict, tagged_file_path + file)
    FileIterator.dump_array(another_file if output_to_another_file else tw_file, tw_arr)


def update_tw_arr_dict_from_tagged_file(tw_arr_dict, tagged_file):
    with open(tagged_file) as fp:
        for line in fp.readlines():
            twid, tag = 0, 1      # parse lines
            tw_arr_dict[twid][TweetKeys.key_ptagtime] += 1 if tag == 1 else 0
            tw_arr_dict[twid][TweetKeys.key_ntagtime] += 1 if tag == -1 else 0
            tw_arr_dict[twid][TweetKeys.key_tagtimes] += 1


def exec_train(seed_parser, unlb_parser, cntr_parser):
    localcounter = WordFreqCounter()
    efe = EventFeatureExtractor()
    efe.load_seed_twarr(seed_parser.get_query_result_file_name())
    efe.load_cntr_twarr(cntr_parser.get_query_result_file_name())
    localcounter, event_classifier = efe.train_and_test(localcounter, None)
    # localcounter.dump_worddict(seed_parser.get_dict_file_name())
    # event_classifier.save_params(seed_parser.get_param_file_name())


# def temp(parser):
#     twarr = FileIterator.load_array(parser.get_query_result_file_name())
#     print(parser.get_query_result_file_name())
#     pos_dict = {}
#     for tw in twarr:
#         wordlabels = tw[TweetKeys.key_wordlabels]
#         for wordlabel in wordlabels:
#             pos_dict[wordlabel[2]] = {}
#     for idx, k in enumerate(sorted(pos_dict.keys())):
#         print(idx, k)
#     print('dict length:', len(pos_dict.keys()))
#     FileIterator.dump_array('posdict.txt', [pos_dict], sort_keys=True)


def exec_query_unlabelled(data_path, parser):
    efe = EventFeatureExtractor()
    efe.start_ner_service()
    # Actually only unlb_parser should use this function
    data_path = FileIterator.append_slash_if_necessary(data_path)
    file_list = [file_name for file_name in FileIterator.listchildren(data_path, children_type='file') if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    date_group = ArrayUtils.group_array_by_condition(file_list, lambda item: item[:11])
    file_list = [[(data_path + file_name) for file_name in group] for group in date_group]
    
    start = 0
    process_num = 16
    twarr_block = list()
    while True:
        added_twarr_block = FileIterator.multi_process(query_tw_file_multi,
            [(file_slice, parser.__class__, parser.query_list, parser.theme, parser.description)
             for file_slice in file_list[start * process_num: (start + 1) * process_num]])
        twarr_block.extend(added_twarr_block)
        if (start + 1) * process_num > len(file_list):
            break
        start += 1
    
    for twarr in twarr_block:
        efe.twarr_ner(twarr)
    
    # print('len(parser.added_twarr)', len(parser.added_twarr))
    # remove_similar_tws(parser.added_twarr)
    # print('accepted parser.added_twarr', len(parser.added_twarr))
    # FileIterator.dump_array(parser.get_query_result_file_name(), parser.added_twarr)
    
    efe.end_ner_service()

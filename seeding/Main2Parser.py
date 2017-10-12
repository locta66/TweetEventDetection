import FileIterator
import TweetKeys
from EventFeatureExtractor import EventFeatureExtractor
import Levenshtein


def exec_query(data_path, parser):
    data_path = FileIterator.append_slash_if_necessary(data_path)
    FileIterator.iterate_file_tree(data_path, query_tw_files_in_path_multi, parser=parser)
    print('Totally', len(parser.added_twarr), 'tweets queried.')
    remove_similar_tws(parser.added_twarr)
    print(parser.get_query_result_file_name(), 'written.\n', len(parser.added_twarr), 'tweets accepted.')
    FileIterator.dump_array(parser.get_query_result_file_name(), parser.added_twarr)


def query_tw_files_in_path_multi(json_path, *args, **kwargs):
    parser = kwargs['parser']
    subfiles = FileIterator.listchildren(json_path, children_type='file')
    file_list = [(json_path + file_name) for file_name in subfiles if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    print(file_list[0])
    print(len(file_list), 'files until')
    print(file_list[-1])
    file_list = FileIterator.split_into_multi_format(file_list, process_num=16)
    added_twarr_block = FileIterator.multi_process(query_tw_file_multi,
        [(file_list_slice, parser.__class__, parser.query_list, parser.theme,
          parser.description) for file_list_slice in file_list])
    parser.added_twarr = FileIterator.merge_list(added_twarr_block)


def query_tw_file_multi(file_list, parser_class, query_list, theme, description):
    parser = parser_class(query_list, theme, description)
    for file in file_list:
        parser.read_tweet_from_json_file(file)
    return parser.added_twarr


def remove_similar_tws(twarr, sim_threashold=0):
    key_cleantext = TweetKeys.key_cleantext
    for i in range(len(twarr)-1, -1, -1):
        for j in range(len(twarr)-1, i, -1):
            dist = Levenshtein.distance(twarr[i][key_cleantext], twarr[j][key_cleantext]) + 1
            if max(len(twarr[i][key_cleantext]), len(twarr[j][key_cleantext])) / dist > 6:
                print(twarr[j])
                del twarr[j]
                break
    # tw_id_cpy = sorted([(idx, tw) for idx, tw in enumerate(twarr)], key=lambda item: item[1][key_cleantext])
    # print(tw_id_cpy, '\n')
    # prev = 0
    # remove_ids = list()
    # for cur in range(1, len(tw_id_cpy)):
    #     dist = Levenshtein.distance(tw_id_cpy[prev][1][key_cleantext], tw_id_cpy[cur][1][key_cleantext]) + 1
    #     print(tw_id_cpy[prev][1][key_cleantext], '\n', tw_id_cpy[cur][1][key_cleantext])
    #     print(max(len(tw_id_cpy[prev][1][key_cleantext]), len(tw_id_cpy[cur][1][key_cleantext])) / dist, '\n')
    #     if max(len(tw_id_cpy[prev][1][key_cleantext]), len(tw_id_cpy[cur][1][key_cleantext])) / dist > 6:
    #         remove_ids.append(tw_id_cpy[cur][0])
    #     else:
    #         prev = cur
    # for idx in sorted(remove_ids, reverse=True):
    #     del twarr[idx]
    return twarr


# def query_tw_files_in_path(json_path, *args, **kwargs):
#     parser = kwargs['parser']
#     subfiles = FileIterator.listchildren(json_path, children_type='file')
#     for subfile in subfiles[0:72]:
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


def update_tw_arr_dict_from_tagged_file(tw_arr_dict, tagged_file):
    with open(tagged_file) as fp:
        for line in fp.readlines():
            twid, tag = 0, 1      # parse lines
            tw_arr_dict[twid][TweetKeys.key_ptagtime] += 1 if tag == 1 else 0
            tw_arr_dict[twid][TweetKeys.key_ntagtime] += 1 if tag == -1 else 0
            tw_arr_dict[twid][TweetKeys.key_tagtimes] += 1


def update_tw_file_from_tag_in_path(tw_file, tagged_file_path, tagged_postfix='.tag',
                                    output_to_another_file=False, another_file='./deafult.sum'):
    tw_arr = FileIterator.load_array(tw_file)
    tw_arr_dict = dict()
    for tw in tw_arr:
        tw_arr_dict[tw['id']] = tw
    subfiles = FileIterator.listchildren(tagged_file_path, children_type='file')
    for subfile in subfiles:
        if not subfile.endswith(tagged_postfix):
            continue
        tagged_file = tagged_file_path + subfile
        update_tw_arr_dict_from_tagged_file(tw_arr_dict, tagged_file)
    output_file = another_file if output_to_another_file else tw_file
    FileIterator.dump_array(output_file, tw_arr)


def exec_untag(parser):
    update_tw_file_from_tag_in_path(parser.get_query_result_file_name(), parser.get_theme_path())

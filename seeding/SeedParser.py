import FileIterator
from JsonParser import JsonParser
from SeedQuery import SeedQuery
from EventFeatureExtractor import EventFeatureExtractor


class SeedParser(JsonParser):
    key_ptagtime = 'ptt'
    key_ntagtime = 'ntt'
    key_tagtimes = 'ttms'
    
    def __init__(self, query_list, theme, description, is_seed):
        JsonParser.__init__(self)
        self.base_path = './'
        self.theme = theme
        self.description = description
        self.is_seed = is_seed
        
        self.query_list = query_list
        self.seed_query_list = [SeedQuery(*query) for query in query_list]
        self.tweet_desired_attrs.append('norm')
        
        self.added_ids = dict()
        self.added_twarr = list()
    
    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        with open(file) as fp:
            for line in fp.readlines():
                tw = self.parse_text(line, filtering=False)
                tw = self.attribute_filter(tw, self.tweet_desired_attrs)
                if 'user' in tw:
                    tw['user'] = self.attribute_filter(tw['user'], self.user_desired_attrs)
                tw_added = False
                for seed_query in self.seed_query_list:
                    tw_added = seed_query.append_desired_tweet(tw, usingtwtime=False) or tw_added
                if tw_added:
                    if tw['id'] in self.added_ids:
                        continue
                    else:
                        self.added_ids[tw['id']] = True
                    self.added_twarr.append(tw)
    
    def is_file_of_query_date(self, file_name):
        for seed_query in self.seed_query_list:
            tw_ymd = seed_query.time_of_tweet(file_name, source='filename')
            if seed_query.is_time_desired(tw_ymd):
                return True
        return False
    
    def set_base_path(self, base_path):
        self.base_path = FileIterator.append_slash_if_necessary(base_path)
        FileIterator.make_dirs_if_not_exists(self.get_base_path())
        FileIterator.make_dirs_if_not_exists(self.get_theme_path())
        FileIterator.make_dirs_if_not_exists(self.get_queried_path())
        FileIterator.make_dirs_if_not_exists(self.get_param_path())
        FileIterator.make_dirs_if_not_exists(self.get_dict_path())
    
    def get_base_path(self):
        return self.base_path
    
    def get_theme_path(self, base_path=None):
        base_path = base_path if base_path else self.get_base_path()
        return FileIterator.append_slash_if_necessary(base_path + self.theme)
    
    def get_queried_path(self):
        return FileIterator.append_slash_if_necessary(self.get_theme_path() + 'queried')
    
    def get_param_path(self):
        return FileIterator.append_slash_if_necessary(self.get_theme_path() + 'params')
    
    def get_dict_path(self):
        return FileIterator.append_slash_if_necessary(self.get_theme_path() + 'dict')
    
    def get_query_result_file_name(self):
        return self.get_queried_path() + self.theme + '%s.sum' % ('' if self.is_seed else '_unlabeled')
    
    def get_to_tag_file_name(self):
        return self.get_queried_path() + self.theme + '%s.utg' % ('' if self.is_seed else '_unlabeled')
    
    def get_param_file_name(self):
        return self.get_param_path() + self.theme + '.prm'
    
    def get_dict_file_name(self):
        return self.get_param_path() + self.theme + '.dic'


def query_tw_file_multi(file_list, query_list, theme, description, is_seed):
    parser = SeedParser(query_list, theme, description, is_seed)
    for file in file_list:
        parser.read_tweet_from_json_file(file)
    return parser.added_twarr


def query_tw_files_in_path_multi(json_path, *args, **kwargs):
    parser = kwargs['parser']
    subfiles = FileIterator.listchildren(json_path, children_type='file')
    file_list = [(json_path + file_name) for file_name in subfiles if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    file_list = FileIterator.split_into_multi_format(file_list, process_num=16)
    added_twarr_block = FileIterator.multi_process(query_tw_file_multi,
        [(file_list_slice, parser.query_list, parser.theme,
          parser.description, parser.is_seed, ) for file_list_slice in file_list])
    parser.added_twarr.extend(FileIterator.merge_list(added_twarr_block))


# def query_tw_files_in_path(json_path, *args, **kwargs):
#     parser = kwargs['parser']
#     subfiles = FileIterator.listchildren(json_path, children_type='file')
#     for subfile in subfiles[0:72]:
#         if not subfile.endswith('.sum'):
#             continue
#         json_file = json_path + subfile
#         parser.read_tweet_from_json_file(json_file)


def exec_query(data_path, parser):
    data_path = FileIterator.append_slash_if_necessary(data_path)
    FileIterator.iterate_file_tree(data_path, query_tw_files_in_path_multi, parser=parser)
    print(parser.get_query_result_file_name(), 'written')
    print('added_twarr num:', len(parser.added_twarr))
    FileIterator.dump_array(parser.get_query_result_file_name(), parser.added_twarr)


def reset_tag_of_tw_file(tw_file):
    tw_arr = FileIterator.load_array(tw_file)
    for tw in tw_arr:
        tw[SeedParser.key_ptagtime] = 0
        tw[SeedParser.key_ntagtime] = 0
        tw[SeedParser.key_tagtimes] = 0
    FileIterator.dump_array(tw_file, tw_arr)


def exec_totag(parser):
    reset_tag_of_tw_file(parser.get_query_result_file_name())
    tw_file = parser.get_query_result_file_name()
    to_tag_file = parser.get_to_tag_file_name()
    to_tag_dict = dict()
    for tw in FileIterator.load_array(tw_file):
        to_tag_dict[tw['id']] = tw['norm']
    FileIterator.dump_array(to_tag_file, [parser.theme], overwrite=True)
    FileIterator.dump_array(to_tag_file, [parser.description], overwrite=False)
    FileIterator.dump_array(to_tag_file, [parser.get_theme_path()], overwrite=False)
    FileIterator.dump_array(to_tag_file, [to_tag_dict], overwrite=False)


def exec_ner(parser):
    efe = EventFeatureExtractor()
    efe.start_ner_service()
    efe.perform_ner_on_tw_file(parser.get_query_result_file_name())
    efe.end_ner_service()


def update_tw_arr_dict_from_tagged_file(tw_arr_dict, tagged_file):
    with open(tagged_file) as fp:
        for line in fp.readlines():
            twid, tag = 0, 1      # parse lines
            tw_arr_dict[twid][SeedParser.key_ptagtime] += 1 if tag == 1 else 0
            tw_arr_dict[twid][SeedParser.key_ntagtime] += 1 if tag == -1 else 0
            tw_arr_dict[twid][SeedParser.key_tagtimes] += 1


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

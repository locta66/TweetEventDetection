import os
import shutil

import FileIterator
from JsonParser import JsonParser
from SeedQuery import SeedQuery
from EventFeatureExtractor import EventFeatureExtractor


class SeedParser(JsonParser):
    def __init__(self, query_list, theme, description):
        JsonParser.__init__(self)
        self.theme = theme
        self.description = description
        self.query_list = query_list
        self.seed_query_list = [SeedQuery(*query) for query in self.query_list]
        self.tweet_desired_attrs = ['created_at', 'id', 'text', 'norm', 'place',
                                    'user', 'retweet_count', 'favorite_count']
        self.added_ids = {}
        self.added_twarr = list()
    
    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        with open(file) as json_file:
            for line in json_file.readlines():
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
    
    def get_theme_path(self, base_path):
        return FileIterator.append_slash_if_necessary(base_path + self.theme)
    
    def get_query_result_file_name(self, base_path):
        return self.get_theme_path(base_path) + self.theme + '.sum'
    
    def get_to_tag_file_name(self, base_path):
        return self.get_theme_path(base_path) + self.theme + '.utg'
    
    def dump_query_list_results(self, seed_path):
        print(len(self.added_twarr))
        theme_path = self.get_theme_path(seed_path)
        if os.path.exists(theme_path):
            shutil.rmtree(theme_path)
        os.makedirs(theme_path)
        FileIterator.dump_array(self.get_query_result_file_name(seed_path), self.added_twarr)
    
    def create_to_tag_form_file_of_query(self, tw_file, to_tag_file):
        to_tag_dict = {}
        for tw in FileIterator.load_array(tw_file):
            to_tag_dict[tw['id']] = tw['norm']
        print(len(to_tag_dict.keys()))
        FileIterator.dump_array(to_tag_file, [self.theme], overwrite=True)
        FileIterator.dump_array(to_tag_file, [self.description], overwrite=False)
        directory = FileIterator.append_slash_if_necessary(os.path.dirname(tw_file))
        FileIterator.dump_array(to_tag_file, [directory], overwrite=False)
        FileIterator.dump_array(to_tag_file, [to_tag_dict], overwrite=False)


def parse_files_in_path(json_path, *args, **kwargs):
    seed_parser = kwargs['seed_parser']
    subfiles = FileIterator.listchildren(json_path, children_type='file')
    for subfile in subfiles[0:15]:
        if not subfile.endswith('.sum'):
            continue
        json_file = json_path + subfile
        seed_parser.read_tweet_from_json_file(json_file)


since = ['2016', '11', '01']
until = ['2016', '11', '30']
seed_parser = SeedParser([
    [{'any_of': ['terror', 'attack']}, since, until],
], theme='Terrorist', description='Event of terrorist attack')


def parse_querys(data_path, seed_path):
    data_path = FileIterator.append_slash_if_necessary(data_path)
    FileIterator.iterate_file_tree(data_path, parse_files_in_path, seed_parser=seed_parser)
    seed_parser.dump_query_list_results(seed_path)


def create_to_tag(seed_path):
    seed_parser.create_to_tag_form_file_of_query(seed_parser.get_query_result_file_name(seed_path),
                                                 seed_parser.get_to_tag_file_name(seed_path))


def perform_ner(seed_path):
    efe = EventFeatureExtractor()
    efe.perform_ner_on_tw_file(seed_parser.get_query_result_file_name(seed_path))

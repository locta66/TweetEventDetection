import os
import shutil

import FileIterator
from SeedParser import SeedParser
from NerServiceProxy import get_ner_service_pool


class TweetTagger(SeedParser):
    def __init__(self, query_list, theme):
        SeedParser.__init__(self, query_list, theme)
    
    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        with open(file) as json_file:
            for line in json_file.readlines():
                tw = self.parse_text(line, filtering=False)
                tw_af = self.attribute_filter(tw, self.tweet_desired_attrs)
                if 'user' in tw_af:
                    tw_af['user'] = self.attribute_filter(tw_af['user'], self.user_desired_attrs)
                tw_added = False
                for seed_query in self.seed_query_list:
                    tw_added = seed_query.append_desired_tweet(tw_af, usingtwtime=False) or tw_added
                if tw_added:
                    self.added_twarr.append(tw_af)
    
    def dump_query_list_results(self, seed_path):
        self.twarr_ner(self.added_twarr)
        for tw in self.added_twarr:
            tw['ltimes'] = 0
            tw['ptimes'] = 0
        theme = self.theme if self.theme else 'default_theme'
        theme_path = FileIterator.append_slash_if_necessary(seed_path + theme)
        tw_file = theme_path + theme + '.sum'
        FileIterator.dump_array(tw_file, self.added_twarr)
    
    def create_untagged_form_tw_file(self, tw_file, untagged_file):
        to_tag_arr = []
        tw_arr = FileIterator.load_array(tw_file)
        for tw in tw_arr:
            tag_content = dict({'id': tw['id'], 'text': tw['norm']})
            to_tag_arr.append(tag_content)
        FileIterator.dump_array(untagged_file, to_tag_arr)
    
    def update_tw_file_from_tagged_file(self, tw_file, tagged_file):
        tw_arr = FileIterator.load_array(tw_file)
        temp_dict = {}
        for tw in tw_arr:
            temp_dict[tw['id']] = tw
        tagged_arr = FileIterator.load_array(tagged_file)
        for tagged in tagged_arr:
            if tagged['id'] in temp_dict:
                temp_dict[tagged['id']]['label'] = tagged['label']
        FileIterator.dump_array(tw_file, tw_arr)


def parse_files_in_path(json_path, *args, **kwargs):
    tw_tagger = kwargs['tw_tagger']
    subfiles = FileIterator.listchildren(json_path, children_type='file')
    for subfile in subfiles:
        if not subfile.endswith('.sum'):
            continue
        json_file = json_path + subfile
        tw_tagger.read_tweet_from_json_file(json_file)


def parse_querys(data_path, seed_path):
    get_ner_service_pool().start(pool_size=8, classify=False, pos=True)
    since = ['2016', '11', '01']
    until = ['2016', '11', '30']
    tw_tagger = TweetTagger([
        [{'any_of': ['terror', 'attack', 'isis']}, since, until],
    ], theme='Terrorist')
    data_path = FileIterator.append_slash_if_necessary(data_path)
    FileIterator.iterate_file_tree(data_path, parse_files_in_path, tw_tagger=tw_tagger)
    get_ner_service_pool().end()

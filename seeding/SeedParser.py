import FileIterator
from JsonParser import JsonParser
from SeedQuery import SeedQuery
import TweetKeys


class SeedParser(JsonParser):
    def __init__(self, query_list, theme, description):
        JsonParser.__init__(self)
        self.base_path = './'
        self.theme = theme
        self.description = description
        
        self.query_list = query_list
        self.seed_query_list = [SeedQuery(*query) for query in query_list]
        self.tweet_desired_attrs.append(TweetKeys.key_origintext)
        
        self.added_ids = dict()
        self.added_twarr = list()
    
    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        for tw in FileIterator.load_array(file):
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
        for path in [self.get_base_path(), self.get_theme_path(), self.get_queried_path(),
                     self.get_param_path(), self.get_dict_path()]:
            FileIterator.make_dirs_if_not_exists(path)
    
    def get_base_path(self):
        return self.base_path
    
    def get_theme_path(self):
        return FileIterator.append_slash_if_necessary(self.get_base_path() + self.theme)
    
    def get_queried_path(self):
        return FileIterator.append_slash_if_necessary(self.get_theme_path() + 'queried')
    
    def get_param_path(self):
        return FileIterator.append_slash_if_necessary(self.get_theme_path() + 'params')
    
    def get_dict_path(self):
        return FileIterator.append_slash_if_necessary(self.get_theme_path() + 'dict')
    
    def get_query_result_file_name(self):
        return self.get_queried_path() + self.theme + '.sum'
    
    def get_to_tag_file_name(self):
        return self.get_queried_path() + self.theme + '.utg'
    
    def get_param_file_name(self):
        return self.get_param_path() + self.theme + '.prm'
    
    def get_dict_file_name(self):
        return self.get_dict_path() + self.theme + '.dic'


class UnlbParser(SeedParser):
    def __init__(self, query_list, theme, description):
        SeedParser.__init__(self, query_list, theme, description)

    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        for tw in FileIterator.load_array(file):
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
                if len(self.added_twarr) > 200:
                    return
    
    def get_query_result_file_name(self):
        return self.get_queried_path() + self.theme + '_unlabeled.sum'
    
    def get_to_tag_file_name(self):
        raise ValueError('Unimplemented yet')


class CounterParser(SeedParser):
    def __init__(self, query_list, theme, description):
        SeedParser.__init__(self, query_list, theme, description)
    
    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        for tw in FileIterator.load_array(file):
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
                if len(self.added_twarr) > 200:
                    return
    
    def get_query_result_file_name(self):
        return self.get_queried_path() + self.theme + '_counter.sum'
    
    def get_to_tag_file_name(self):
        raise ValueError('Unimplemented yet')

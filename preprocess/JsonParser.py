import os
import re
import json
import bz2file

from Pattern import get_pattern
from TweetDict import TweetDict


class JsonParser:
    def __init__(self, tweet_desired_attrs=None, user_desired_attrs=None):
        self.tweet_desired_attrs = ['created_at', 'id', 'text', 'place',    # 'geo', 'coordinates',
                                    'user', 'retweet_count', 'favorite_count']
        self.user_desired_attrs = ['id', 'time_zone', 'location', 'followers_count',
                                   'friends_count', 'listed_count', 'statuses_count']
        self.pattern = get_pattern()
        self.twdict = TweetDict()
    
    def parse_text(self, text, filtering):
        json_obj = json.loads(text)
        if filtering:
            return self.tweet_filter(json_obj)
        else:
            return json_obj
    
    def read_tweet_from_bz2_file(self, file, filtering=True):
        fp = bz2file.open(file, 'r')
        tw_arr = []
        for line in fp.readlines():
            line = line.decode('utf8')
            tw = self.parse_text(line, filtering)
            tw_arr.extend([tw] if tw else [])
        fp.close()
        return tw_arr
    
    def read_tweet_from_json_file(self, file, filtering=True):
        """
        Given file with lines of json format string, returns tweets converted from those lines.
        :param file:
        :param filtering:
        :return: List of tweets of interest.
        """
        with open(file) as json_file:
            tw_arr = []
            for line in json_file.readlines():
                tw = self.parse_text(line, filtering)
                tw_arr.extend([tw] if tw else [])
            return tw_arr
    
    def tweet_filter(self, tw):
        if not('text' in tw and tw['lang'] == 'en'):
            return None
        tw_filtered = self.attribute_filter(tw, self.tweet_desired_attrs)
        if 'user' in tw_filtered:
            tw_filtered['user'] = self.attribute_filter(tw_filtered['user'], self.user_desired_attrs)
        if 'text' in tw_filtered:
            normalized_text = self.pattern.normalization(tw_filtered['text'])
            if len(re.split('\s', normalized_text)) <= 3:
                return None
            tw_filtered['norm'] = normalized_text
            tw_filtered['text'] = self.sentence_filter(normalized_text)
        return tw_filtered
    
    def sentence_filter(self, text):
        sentences = re.split('[.?!]', text)
        res = []
        for sentence in sentences:
            if sentence is '':
                continue
            sentence_seg = self.twdict.text_regularization(sentence, seglen=12)
            res.append(' '.join(sentence_seg))
        return ' . '.join(res)
    
    def attribute_filter(self, target_dict, attr_list):
        for attr in list(target_dict.keys()):
            if attr not in attr_list:
                target_dict.pop(attr)
        # extracted_dict = {}
        # for attr in attr_list:
        #     extracted_dict[attr] = target_dict[attr]
        return target_dict
    
    @staticmethod
    def dump_json_arr_into_file(json_arr, file_path, mode='append'):
        if mode is 'append':
            pass
        elif mode is 'renew':
            if os.path.exists(file_path):
                os.remove(file_path)
        with open(file_path, 'a') as fp:
            for obj in json_arr:
                line = json.dumps(obj, sort_keys=True) + '\n'
                fp.write(line)


# ######## 推文属性 ########
# text

# created_at
# timestamp_ms
# place
# geo
# coordinates

# lang
# filter_level
# truncated
# is_quote_status
# entities
# retweeted
# retweet_count
# favorited
# favorite_count
# user
# id
# in_reply_to_user_id
# in_reply_to_user_id_str
# in_reply_to_status_id
# in_reply_to_status_id_str
# in_reply_to_screen_name
# ######## 推文属性 undesired ########
# contributors
# source
# id_str


# ######## 用户属性 ########
# id
# id_str
# name
# time_zone
# lang
# location
# geo_enabled

# followers_count
# friends_count
# listed_count
# statuses_count
# screen_name

# notifications
# contributors_enabled
# protected
# utc_offset
# url
# verified
# is_translator
# description
# ######## 用户属性 undesired ########
# created_at
# following
# follow_request_sent
# favourites_count

# default_profile
# profile_link_color
# profile_text_color
# profile_sidebar_fill_color
# profile_background_color
# profile_sidebar_border_color
# default_profile_image
# profile_image_url
# profile_image_url_https
# profile_background_tile
# profile_use_background_image
# profile_background_image_url
# profile_background_image_url_https

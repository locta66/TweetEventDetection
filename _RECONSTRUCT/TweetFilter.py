import re
import TweetKeys as tk
import PatternUtils as pu
from Pattern import get_pattern
from MyDict import MyDict

FILTER_LEVEL_LOW = 'low'
FILTER_LEVEL_HIGH = 'high'

low_tw_attr = ['created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
                       'retweet_count', 'favorite_count', 'entities', 'source',
                       'filter_level', 'truncated', 'is_quote_status',
                       'in_reply_to_status_id', 'in_reply_to_user_id']
low_user_attr = ['id', 'created_at', 'time_zone', 'location', 'favourites_count',
                         'followers_count', 'friends_count', 'listed_count', 'statuses_count',
                         'contributors_enabled', 'protected', 'is_translator',
                         'description', 'verified']

high_tw_attr = ['created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
                        'retweet_count', 'favorite_count', 'in_reply_to_status_id', 'in_reply_to_user_id']
high_user_attr = ['id', 'created_at', 'time_zone', 'location', 'favourites_count',
                          'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'verified']

attr_dict = {'low': [low_tw_attr, low_user_attr], 'high': [high_tw_attr, high_user_attr]}
mydict = MyDict()


def filter_twarr(twarr, filter_level='low'):
    tweet_desired_attrs, user_desired_attrs = attr_dict[filter_level]
    removal_idx = list()
    for idx, tw in enumerate(twarr):
        if tk.key_lang not in tw or tk.key_text not in tw or not tw[tk.key_lang] == 'en':
            removal_idx.append(idx)
            continue
        tw = attribute_filter(tw, tweet_desired_attrs)
        if tk.key_user in tw:
            tw[tk.key_user] = attribute_filter(tw[tk.key_user], user_desired_attrs)
        if tk.key_text in tw:
            tw[tk.key_origintext] = tw[tk.key_text]
            normalized_text = get_pattern().normalization(tw[tk.key_text])
            if len(re.split('[\s,.!?]', normalized_text)) <= 5:
                removal_idx.append(idx)
                continue
            tw[tk.key_text] = sentence_filter(normalized_text)
    for idx in removal_idx[::-1]:
        del twarr[idx]
    return twarr


def eliminate_dup_id(twarr):
    id_set, dup_idx = set(), set()
    for idx, tw in enumerate(twarr):
        tw_id = tw[tk.key_id]
        if tw_id not in id_set:
            id_set.add(tw_id)
        else:
            dup_idx.add(idx)
    for idx in range(len(twarr)-1, -1, -1):
        if idx in dup_idx:
            del twarr[idx]
    return twarr


def attribute_filter(target_dict, attr_list):
    for attr in list(target_dict.keys()):
        if attr not in attr_list:
            target_dict.pop(attr)
    return target_dict


def text_filter(text):
    normalized_text = get_pattern().normalization(text)
    return sentence_filter(normalized_text)


def sentence_filter(text):
    sentences = re.split('[.?!]', text)
    res = list()
    for sentence in sentences:
        if sentence is not '':
            sentence_seg = mydict.text_regularization(sentence)
            res.append(' '.join(sentence_seg))
    return ' . '.join(res)

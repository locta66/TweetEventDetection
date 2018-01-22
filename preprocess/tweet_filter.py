import utils.tweet_keys as tk
import utils.pattern_utils as pu


FILTER_LEVEL_LOW = 'low'
FILTER_LEVEL_HIGH = 'high'
FILTER_LEVEL_NONE = 'none'

low_tw_attr = {'created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
               'retweet_count', 'favorite_count', 'entities', 'source',
               'filter_level', 'truncated', 'is_quote_status',
               'in_reply_to_status_id', 'in_reply_to_user_id'}
low_user_attr = {'id', 'created_at', 'time_zone', 'location', 'favourites_count',
                 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
                 'contributors_enabled', 'protected', 'is_translator',
                 'description', 'verified'}

high_tw_attr = {'created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
                'retweet_count', 'favorite_count', 'in_reply_to_status_id', 'in_reply_to_user_id'}
high_user_attr = {'id', 'created_at', 'time_zone', 'location', 'favourites_count',
                  'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'verified'}

attr_dict = {FILTER_LEVEL_LOW: (low_tw_attr, low_user_attr),
             FILTER_LEVEL_HIGH: (high_tw_attr, high_user_attr),
             FILTER_LEVEL_NONE: (None, None)}


def filter_twarr(twarr, attr_filt=True, text_filt=True, filter_level=FILTER_LEVEL_LOW):
    """ may result in change in the length of twarr """
    _tw_attrs, _usr_attrs = attr_dict[filter_level]
    _removal_idx = list()
    
    _attr_filter = attr_filt if callable(attr_filt) else attribute_filter if attr_filt else None
    _text_filter = text_filt if callable(text_filt) else text_filter if text_filt else None
    
    for _idx, tw in enumerate(twarr):
        if tk.key_lang not in tw or not tw[tk.key_lang] == 'en' or tk.key_text not in tw:
            _removal_idx.append(_idx)
            continue
        """ filter attributes """
        if _attr_filter is not None:
            if _attr_filter is attribute_filter:
                tw = _attr_filter(tw, _tw_attrs)
                if tk.key_user in tw:
                    tw[tk.key_user] = _attr_filter(tw[tk.key_user], _usr_attrs)
            else:
                _attr_filter(tw)
        """ filter text content """
        if tk.key_text in tw and _text_filter is not None:
            if _text_filter is text_filter:
                tw[tk.key_origintext] = tw[tk.key_text]
                normalized_text = text_filter(tw[tk.key_text])
                if len(pu.tokenize(r'[a-zA-Z_\-\']{3,}', normalized_text)) <= 5:
                    _removal_idx.append(_idx)
                    continue
                tw[tk.key_text] = normalized_text
            else:
                _text_filter(tw)
    
    for _idx in _removal_idx[::-1]:
        del twarr[_idx]
    return twarr


def filter_twarr_attr(twarr,
                      tw_cond=lambda tw: not(tk.key_lang in tw and tw[tk.key_lang] == 'en' and tk.key_text in tw),
                      attr_filter=None, filter_level=FILTER_LEVEL_LOW):
    if attr_filter is not None:
        for tw in twarr:
            attr_filter(tw)
        return twarr
    else:
        res_twarr = []
        tw_attrs, usr_attrs = attr_dict[filter_level]
        for tw in twarr:
            if tw_cond(tw):
                tw = attribute_filter(tw, tw_attrs)
                if tk.key_user in tw:
                    tw[tk.key_user] = attribute_filter(tw[tk.key_user], usr_attrs)
                res_twarr.append(tw)
        return res_twarr


def filter_twarr_text(twarr,
                      tw_cond=lambda tw: tk.key_text in tw,
                      get_text=lambda tw: tw.get(tk.key_text),
                      flt_text=lambda text: pu.text_normalization(text),
                      set_text=lambda tw, text: tw.setdefault(tk.key_text, text)):
    for _tw in twarr:
        if tw_cond(_tw):
            text = get_text(_tw)
            text = flt_text(text)
            set_text(_tw, text)
    return twarr


def filter_twarr_dup_id(twarr, get_id=lambda tw: tw.get(tk.key_id)):
    id_set, dup_idx_list = set(), list()
    for _idx, _tw in enumerate(twarr):
        tw_id = get_id(_tw)
        if tw_id not in id_set:
            id_set.add(tw_id)
        else:
            dup_idx_list.append(_idx)
    for _idx in range(len(dup_idx_list)-1, -1, -1):
        del twarr[dup_idx_list[_idx]]
    return twarr


def attribute_filter(target_dict, attr_set):
    if attr_set is None:
        return target_dict
    for _tw_attr in set(target_dict.keys()):
        if _tw_attr not in attr_set:
            target_dict.pop(_tw_attr)
    return target_dict


def text_filter(text):
    return pu.text_normalization(text)

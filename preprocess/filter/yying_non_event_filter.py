import re
import pickle
import datetime
import traceback

import pandas as pd

from config.configure import getcfg
from preprocess.filter.filter_utils import get_all_substrings
from preprocess.filter.pos_tag_process import count_sentiment

import utils.array_utils as au
import utils.function_utils as fu
import utils.file_iterator as fi
import utils.pattern_utils as pu
import utils.tweet_keys as tk
import utils.timer_utils as tmu


chat_filter_file = getcfg().chat_filter_file
is_noise_dict_file = getcfg().is_noise_dict_file

clf_model_file = getcfg().clf_model_file
black_list_file = getcfg().black_list_file


class UseGSDMM:
    def __init__(self):
        try:
            with open(chat_filter_file, 'rb') as f:
                self.c = pickle.load(f)
            with open(is_noise_dict_file, 'rb') as f:
                self.is_noise_dict = set(pickle.load(f))
        except:
            print('load error')
            traceback.print_exc()

    def get_GSDMM_Feature(self, json):
        """ is tw a chat or not """
        if tk.key_orgntext in json:
            json[tk.key_text] = pu.text_normalization(json[tk.key_orgntext])
        else:
            text = json[tk.key_text]
            json[tk.key_orgntext] = text
            json[tk.key_text] = pu.text_normalization(text)
        topic_num = self.c.sample_cluster(json)
        if topic_num in self.is_noise_dict:
            return 1
        return 0


class EffectCheck:
    def __init__(self):
        self.gsdmm = UseGSDMM()
        with open(black_list_file, 'r') as fp:
            self.spam_words = set([line.strip() for line in fp.readlines()])

    def filter(self, twarr):
        vecarr = [self.get_features(json) for json in twarr]
        try:
            with open(clf_model_file, 'rb') as f:
                clf = pickle.load(f)
        except:
            traceback.print_exc()
            return
        predict = clf.predict_proba(vecarr)
        flt_twarr = list()
        for idx, p in enumerate(predict):
            if p:
                flt_twarr.append(twarr[idx])
        return flt_twarr
    
    def get_features(self, json):
        user = json[tk.key_user]
        if tk.key_description in user and user[tk.key_description] is not None:
            l_profile_description = len(user[tk.key_description])
        else:
            l_profile_description = 0
        FI = user[tk.key_friends_count]
        FE = user[tk.key_followers_count]
        num_tweet_posted = user[tk.key_statuses_count]

        tw_time = json[tk.key_created_at]
        user_born_time = json[tk.key_user][tk.key_created_at]
        # TODO 有些推文时间字段有误，需要判断处理，比如缺了分秒信息
        tw_d = datetime.datetime.strptime(tw_time, '%a %b %d %H:%M:%S %z %Y')
        user_d = datetime.datetime.strptime(user_born_time, '%a %b %d %H:%M:%S %z %Y')
        time_delta = tw_d - user_d
        AU = time_delta.seconds / 60.0 + time_delta.days * 24
        FE_FI_ratio = 0
        if FI != 0:
            FE_FI_ratio = FE / float(FI)
        reputation = 0
        if (FI + FE) != 0:
            reputation = FE / float(FI + FE)

        following_rate = FI / float(AU)
        tweets_per_day = num_tweet_posted / (AU / 24)
        tweets_per_week = num_tweet_posted / (AU / (24 * 7))

        user_features = [l_profile_description, FI, FE, num_tweet_posted, AU, FE_FI_ratio,
                         reputation, following_rate, tweets_per_day, tweets_per_week]
        """ content features """
        if tk.key_orgntext not in json:
            json[tk.key_orgntext] = json[tk.key_text]
        orgn = json[tk.key_orgntext]
        text = json[tk.key_text]
        words = text.split()
        num_words = len(words)
        num_charater = len(text)
        num_white_space = len(re.findall(r'(\s)', text))
        num_capitalization_word = len(re.findall(r'(\b[A-Z]([a-z])*\b)', text))
        num_capital_per_word = num_capitalization_word / num_words

        max_word_length = 0
        mean_word_length = 0
        assert (len(words) > 0)
        for word in words:
            if len(word) > max_word_length:
                max_word_length = len(word)
                mean_word_length += len(word)

        mean_word_length /= len(words)
        num_exclamation_marks = orgn.count('!')
        num_question_marks = orgn.count('?')
        num_urls = len(json['entities']['urls'])
        num_urls_per_word = num_urls / num_words
        num_hashtags = len(json['entities']['hashtags'])
        num_hashtags_per_word = num_hashtags / num_words
        num_mentions = len(json['entities']['user_mentions'])
        num_mentions_per_word = num_mentions / num_words

        substrings = get_all_substrings(text)
        num_spam_words = 0
        for sub in substrings:
            if sub in self.spam_words:
                num_spam_words += 1
        num_spam_words_per_word = num_spam_words / num_words
        content_features = [num_words, num_charater, num_white_space, num_capitalization_word,
                            num_capital_per_word, max_word_length, mean_word_length,
                            num_exclamation_marks, num_question_marks, num_urls, num_urls_per_word,
                            num_hashtags, num_hashtags_per_word, num_mentions,
                            num_mentions_per_word, num_spam_words, num_spam_words_per_word]
        sentiment_frature = count_sentiment(text)
        chat_feature = self.gsdmm.get_GSDMM_Feature(json)
        total_features = list()
        total_features.extend(user_features)
        total_features.extend(content_features)
        total_features.append(sentiment_frature)
        total_features.append(chat_feature)
        return total_features
    
    def get_filter_res(self, twarr):
        data = [self.get_features(tw) for tw in twarr]
        try:
            with open(clf_model_file, 'rb') as fp:
                clf = pickle.load(fp)
        except:
            traceback.print_exc()
            return
        predict = clf.predict(data)
        table = pd.DataFrame(index={"data"}, columns={'保留', '被过滤'}, data=0)
        for i in range(len(predict)):
            if predict[i] == 1:
                table.loc["data", '保留'] += 1
            else:
                table.loc["data", '被过滤'] += 1
        print(table)
        print('总数：', len(predict), '过滤比例：', table.loc["data"]['被过滤'] / len(predict))


if __name__ == '__main__':
    pos_base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/'
    sub_files = fi.listchildren(pos_base, fi.TYPE_FILE, 'txt$', concat=True)
    pos_twarr = au.merge_array([fu.load_array(file) for file in sub_files])
    print(type(pos_twarr), len(pos_twarr))
    
    neg_file = '/home/nfs/yying/data/crawlTwitter/Crawler1/test.json'
    neg_twarr = fu.load_array(neg_file)
    print(type(neg_twarr), len(neg_twarr))
    
    my_filter = EffectCheck()
    
    tmu.check_time()
    my_filter.get_filter_res(pos_twarr)
    tmu.check_time(print_func=lambda dt: print('pos filter time elapsed {}s'.format(dt)))
    
    # tmu.check_time()
    # my_filter.get_filter_res(neg_twarr)
    # tmu.check_time(print_func=lambda dt: print('neg filter time elapsed {}s'.format(dt)))
    
    # print(len(my_filter.filter(data)))

import datetime
import math

import utils.function_utils as fu
import utils.tweet_keys as tk
import utils.file_iterator as fi


class HotDegree:
    def __init__(self):
        self.weight_user = 0.5
        self.weight_time = 0.5
        self.w_propagation = 0.5
        self.weight_tweets_number = 0.001
    
    def hot_degree(self, twarr):
        w_time = self.weight_time
        average_hot = self.calculate_average_hot(twarr)
        duration_time = self.duration(twarr)
        return 0.48 * average_hot + w_time * duration_time + w_time * math.sqrt(len(twarr))
    
    def calculate_average_hot(self, twarr):
        user_influence = propagation_influence = 0
        for idx, tw in enumerate(twarr):
            user = tw[tk.key_user]
            user_influence += self.user_influence(user)
            propagation_influence += self.tw_propagation(tw)
        average_hot = (self.weight_user * user_influence + self.w_propagation * propagation_influence) / \
                      (len(twarr) * 1.0)
        return average_hot

    # TODO 'followers_count'等写到utils.tweet_keys里
    def user_influence(self, user):
        user_i_influence = math.log(user['followers_count'] + 2) * math.log(user['statuses_count'] + 2) * \
                           (1.5 if user['verified'] == 'True' else 1.0)
        return user_i_influence
    
    def tw_propagation(self, tw):
        propagation_i_influence = (0.5 if tw['is_quote_status'] == 'True' else 0) * (tw['retweet_count'] ** 0.5) * 1.0
        return propagation_i_influence
    
    def duration(self, twarr):
        max_distance_hours = 0
        min_distance_hours = 50000000
        for idx, tw in enumerate(twarr):
            year, month, day, hour = self.change_time_format(tw[tk.key_created_at])
            distance = ((datetime.datetime(year, month, day) - datetime.datetime(1970, 1, 1)).days - 1) * 24 + hour
            if distance > max_distance_hours:
                max_distance_hours = distance
            if distance < min_distance_hours:
                min_distance_hours = distance
        return max_distance_hours - min_distance_hours
    
    def change_time_format(self, format_str):
        year = int(format_str[-4:])
        month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7,
                      'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month = month_dict[format_str[4:7]]
        day = int(format_str[8:10])
        hour = int(format_str[11:13])
        return year, month, day, hour


class AttackHot(HotDegree):
    def __int__(self):
        HotDegree.__init__(self)


# TODO 弃用这个函数
def get_every_tweet_status(twarr):
    status_dict = dict()
    for idx, tw in enumerate(twarr):
        status_dict[idx] = {
            'followers_count': tw['user']['followers_count'],
            'statuses_count': tw['user']['statuses_count'],
            'verified': tw['user']['verified'],
            'reply_count': tw['reply_count'] if 'reply_count' in tw else 0,
            'retweet_count': tw['retweet_count'],
            'is_quote_status': tw['is_quote_status'],
            'created_at': tw['created_at']
        }
    return status_dict


if __name__ == '__main__':
    event_files = fi.listchildren('/home/nfs/yangl/event_detection/testdata/event_corpus/', fi.TYPE_FILE,
                                  pattern='.txt$', concat=True)
    model = AttackHot()
    for file in event_files:
        twarr = fu.load_array(file)
        print(model.hot_degree(twarr))

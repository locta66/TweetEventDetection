# import __init__
import datetime


class DateCalculator:
    def __init__(self):
        return
    
    def normalize_ymdh(self, ymdh_str_arr):
        ymd_int_arr = []
        for item in ymdh_str_arr[0:3]:
            if type(item) is str:
                ymd_int_arr.append(item)
            else:
                raise TypeError("Elements of ymdh array is not of valid type str.")
        return ymd_int_arr
    
    # ymdh resembles ['201X', '0X', '2X', '1X'] ,with element of type string
    def is_target_ymdh(self, tweet_ymdh, start_ymdh, end_ymdh):
        if start_ymdh is [] and end_ymdh is []:
            # No ymdh limit given, taken as all dates.
            return True
        if (start_ymdh is not [] or end_ymdh is not []) and tweet_ymdh is []:
            # Once start or end hes been given, then tweet should be provided as well.
            return False
        ds = dt = de = None
        if tweet_ymdh is not []:
            dt = datetime.datetime.strptime('-'.join(self.normalize_ymdh(tweet_ymdh)), '%Y-%m-%d')
        if start_ymdh is not []:
            ds = datetime.datetime.strptime('-'.join(self.normalize_ymdh(start_ymdh)), '%Y-%m-%d')
        if end_ymdh is not []:
            de = datetime.datetime.strptime('-'.join(self.normalize_ymdh(end_ymdh)), '%Y-%m-%d')
        if ds and de and (ds - de).days > 0:
            return False
        is_t_lt_s = (dt - ds).days >= 0 if ds else True
        is_e_lt_t = (de - dt).days >= 0 if de else True
        return is_t_lt_s and is_e_lt_t

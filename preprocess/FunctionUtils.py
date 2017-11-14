import os
import time
import Levenshtein


def slash_appender(func):
    def decorator(*args, **kwargs):
        string = func(*args, **kwargs)
        string = string + os.path.sep if not string.endswith(os.path.sep) else string
        return string
    return decorator


def sync_real_time_counter(info):
    def time_counter(func):
        def decorator(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            print('function name:', func.__name__, ',', info, 'time elapsed:', time.time() - start_time, 's')
        return decorator
    return time_counter


def plagiarize_score(str1, str2):
    if not str1 or not str2:
        return 0
    dist = Levenshtein.distance(str1, str2)
    return 1 - dist / max(len(str1), len(str2))

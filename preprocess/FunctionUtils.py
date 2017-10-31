import os
import time


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
            print('function name:', func.__name__, ',', info, 'time elapsed:', time.time() - start_time)
        return decorator
    return time_counter


def uid_by_random(start=0, length=10):
    import time
    import hashlib
    m = hashlib.md5()
    m.update(bytes(str(time.time()), encoding='utf-8'))
    return m.hexdigest()[start: start + length]

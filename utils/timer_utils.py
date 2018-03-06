import time


default_name = 'main'
check_points = dict()


def check_time(name=default_name, print_func=lambda dt: print('{} s from last check'.format(dt))):
    if name not in check_points:
        check_points[name] = time.time()
        return None
    else:
        delta_t = time.time() - check_points[name]
        if print_func:
            print_func(delta_t)
        check_points[name] = time.time()
        return delta_t

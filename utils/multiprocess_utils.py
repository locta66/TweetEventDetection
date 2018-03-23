import math
import multiprocessing as mp


def split_multi_format(array, process_num):
    block_size = math.ceil(len(array) / process_num)
    formatted_array = list()
    for i in range(process_num):
        arr_slice = array[i * block_size: (i + 1) * block_size]
        if arr_slice:
            formatted_array.append(arr_slice)
    return formatted_array


def multi_process(func, args_list=None, kwargs_list=None):
    """
    Do func in multiprocess way.
    :param func: To be executed within every process
    :param args_list: default () as param for apply_async if not given
    :param kwargs_list:
    :return:
    """
    process_num = len(args_list) if args_list is not None else len(kwargs_list)
    pool = mp.Pool(processes=process_num)
    res_getter = list()
    for i in range(process_num):
        res = pool.apply_async(func=func, args=args_list[i] if args_list else (),
                               kwds=kwargs_list[i] if kwargs_list else {})
        res_getter.append(res)
    pool.close()
    pool.join()
    results = list()
    for i in range(process_num):
        results.append(res_getter[i].get())
    return results


def multi_process_batch(func, batch_size=8, args_list=None, kwargs_list=None):
    total_num = len(args_list)
    if kwargs_list is None:
        kwargs_list = [{} for _ in range(total_num)]
    input_num = 0
    results = list()
    while input_num < total_num:
        since, until = input_num, input_num + batch_size
        res_batch = multi_process(func, args_list[since: until], kwargs_list[since: until])
        results.extend(res_batch)
        input_num += batch_size
    return results


class DaemonPool:
    def __init__(self):
        self.service_on = False
        self.daemon_class = None
        self.daemon_pool = list()
        self._last_task_len = list()
        self._last_batch_len = list()
    
    def is_service_on(self):
        return self.service_on
    
    def start(self, func, pool_size):
        if self.service_on:
            return
        for i in range(pool_size):
            daemon = self.daemon_class(i)
            daemon.start(func)
            self.daemon_pool.append(daemon)
        self.service_on = True
    
    def end(self):
        if not self.service_on:
            return
        for daeprocess in self.daemon_pool:
            daeprocess.end()
        self.daemon_pool.clear()
        self.service_on = False


class DaemonProcess:
    K_END = 'end_process'
    
    def __init__(self, pidx=0):
        self.process = None
        self.pidx = pidx
        self.inq = mp.Queue()
        self.outq = mp.Queue()

    def start(self, func):
        self.process = mp.Process(target=func, args=(self.inq, self.outq))
        self.process.daemon = True
        self.process.start()

    def end(self):
        self.process.terminate()


class CustomDaemonPool(DaemonPool):
    def __init__(self):
        DaemonPool.__init__(self)
        self.daemon_class = CustomDaemonProcess
    
    def set_input(self, arg_list):
        arg_len = len(arg_list)
        self._last_task_len.append(arg_len)
        for idx in range(arg_len):
            daemon = self.daemon_pool[idx]
            args = arg_list[idx]
            daemon.set_input(args)
    
    def get_output(self):
        arg_len = self._last_task_len.pop(0)
        res_list = list()
        for idx in range(arg_len):
            daemon = self.daemon_pool[idx]
            res_list.append(daemon.get_output())
        return res_list
    
    def set_batch_input(self, arg_list):
        # one item in arg_list corresponds to a process
        dpnum = len(self.daemon_pool)
        innum = count = 0
        while innum < len(arg_list):
            self.set_input(arg_list[innum: innum + dpnum])
            innum += dpnum
            count += 1
        self._last_batch_len.append(count)
    
    def get_batch_output(self):
        batch_len = self._last_batch_len.pop(0)
        res_list = list()
        for i in range(batch_len):
            res_list.extend(self.get_output())
        return res_list
    
    """ state of the pool """
    def has_unread_batch_output(self): return len(self._last_batch_len) > 0
    
    def get_unread_batch_output_num(self): return len(self._last_batch_len)
    
    def can_get_batch_output(self):
        if not self.has_unread_batch_output():
            return False
        last_task_len, ready_list = self.get_ready_list()
        return sum(ready_list) == last_task_len
    
    def get_ready_list(self):
        last_task_len = self._last_task_len[0]
        if not self.has_unread_batch_output():
            return last_task_len, [False] * last_task_len
        return last_task_len, [self.daemon_pool[i].outq_size() > 0 for i in range(last_task_len)]


class CustomDaemonProcess(DaemonProcess):
    def __init__(self, pidx=0):
        DaemonProcess.__init__(self, pidx)
        self.unread_task_num = 0
    
    def set_input(self, args):
        self.unread_task_num += 1
        self.inq.put(args)
    
    def get_output(self):
        self.unread_task_num -= 1
        return self.outq.get()
    
    def get_unread_output_num(self): return self.unread_task_num
    
    def has_unread_output(self): return self.unread_task_num > 0
    
    def outq_size(self): return self.outq.qsize()


class ProxyDaemonPool(DaemonPool):
    def __init__(self):
        DaemonPool.__init__(self)
        self.daemon_class = ProxyDaemonProcess
    
    def set_input(self, args_list, kwargs_list):
        arg_len = len(args_list)
        self._last_task_len.append(arg_len)
        for idx in range(arg_len):
            daemon, args, kwargs = self.daemon_pool[idx], args_list[idx], kwargs_list[idx]
            daemon.set_input(args, kwargs)
    
    def get_output(self):
        arg_len = self._last_task_len.pop(0)
        res_list = list()
        for idx in range(arg_len):
            daemon = self.daemon_pool[idx]
            res_list.append(daemon.get_output())
        return res_list
    
    def set_batch_input(self, args_list, kwargs_list=None):
        arg_len = len(args_list)
        if kwargs_list is None:
            kwargs_list = [{} for _ in range(arg_len)]
        dpnum = len(self.daemon_pool)
        innum = count = 0
        while innum < len(args_list):
            self.set_input(args_list[innum: innum + dpnum], kwargs_list[innum: innum + dpnum])
            innum += dpnum
            count += 1
        self._last_batch_len.append(count)
    
    def get_batch_output(self):
        batch_len = self._last_batch_len.pop(0)
        res_list = list()
        for i in range(batch_len):
            res_batch = self.get_output()
            res_list.extend(res_batch)
        return res_list


class ProxyDaemonProcess(DaemonProcess):
    def __init__(self, pidx=0):
        DaemonProcess.__init__(self, pidx)
    
    def start(self, func):
        self.process = mp.Process(target=ProxyDaemonProcess.exec, args=(func, self.inq, self.outq))
        self.process.daemon = True
        self.process.start()
    
    def end(self):
        self.set_input(DaemonProcess.K_END, DaemonProcess.K_END)
        self.process.join()
        self.process.terminate()
    
    def set_input(self, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            # kwargs = {'pidx': self.pidx}
            kwargs = {}
        self.inq.put(args)
        self.inq.put(kwargs)
    
    def get_output(self):
        return self.outq.get()
    
    @staticmethod
    def exec(func, inq, outq):
        while True:
            args = inq.get()
            kwargs = inq.get()
            if args == DaemonProcess.K_END and kwargs == DaemonProcess.K_END:
                break
            result = func(*args, **kwargs)
            outq.put(result)


import utils.function_utils as fu
import utils.file_iterator as fi
import utils.timer_utils as tmu
def read(inq, outq):
    while True:
        idx, file = inq.get()
        twarr = fu.load_array(file)
        outq.put([idx, len(twarr)])
def read2(idx, file, nothing='p'):
    twarr = fu.load_array(file)
    return [idx, len(twarr)]


if __name__ == '__main__':
    # dp = CustomDaemonPool()
    dp = ProxyDaemonPool()
    dp.start(read2, 8)
    # base = '/home/nfs/cdong/tw/testdata/yying/2016_04/'
    # files = [base + sub for sub in subs][:40]
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/'
    subs = fi.listchildren(base, children_type=fi.TYPE_FILE)
    files = [base + sub for sub in subs]
    
    tmu.check_time()
    res = multi_process_batch(read2, args_list=[(idx, file) for idx, file in enumerate(files)])
    # dp.set_batch_input([(idx, file) for idx, file in enumerate(files)],
    #                    [{'nothing': str(idx)+file} for idx, file in enumerate(files)])
    # res = dp.get_batch_output()
    # print(sum(([length for idx, length in res])))
    # print(res)
    tmu.check_time()
    print([[idx, len(fu.load_array(file))] for idx, file in enumerate(files)])
    tmu.check_time()

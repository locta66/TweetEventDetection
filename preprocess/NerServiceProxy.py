import __init__
import math
import subprocess
import multiprocessing as mp
from Configure import getconfig

service_command = getconfig().ner_service_command


class ServiceException(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message


class NerServiceProxy:
    def __init__(self):
        self.service = None
    
    def clear_service(self):
        if self.service is not None:
            self.service.terminate()
            del self.service
            self.service = None
    
    def is_service_open(self):
        return self.service is not None and self.service.poll() is None
    
    def open_ner_service(self, classify, pos):
        if self.is_service_open():
            return
        self.service = subprocess.Popen(service_command, shell=True, close_fds=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = 'open'
        params = '-c' if classify else ''
        params += '-p' if pos else ''
        state, res = self.communicate(command, params)
        if 'success' in state:
            return True
        else:
            self.clear_service()
            return False
    
    def close_ner_service(self):
        if not self.is_service_open():
            return False
        command = 'close'
        params = 'no'
        state, res = self.communicate(command, params)
        if 'success' in state:
            self.clear_service()
            return True
        else:
            self.clear_service()
            return False
    
    def execute_ner(self, text=' '):
        if not self.is_service_open():
            raise ServiceException('Service not open')
        command = 'ex'
        params = text
        state, res = self.communicate(command, params)
        if 'su' == state:
            return res
        else:
            raise ValueError('Error occurs in NER')
    
    # def execute_ner_array(self, textarr):
    #     if not self.is_service_open():
    #         raise ServiceException('Service not open')
    #     resarr = []
    #     num = len(textarr)
    #     self.service.stdin.write(self.append_endline('execute array').encode('utf8'))
    #     self.service.stdin.write(self.append_endline(str(num)).encode('utf8'))
    #     for i in range(num):
    #         self.service.stdin.write(self.append_endline(textarr[i]).encode('utf8'))
    #         self.service.stdin.flush()
    #     for i in range(num):
    #         res = self.service.stdout.readline().decode('utf8').strip()
    #         resarr.append(res)
    #     state = self.service.stdout.readline().decode('utf8').strip()
    #     return resarr
    
    def communicate(self, command, params):
        """
        Invoke proxy to make communication with the process that holds ner extractor.
        :param command: Indicates the operation to be performed by extractor.
        :param params: Params for the operation.
        :return: state: Shows if the operation has been performed properly;
                  res: Contains the NER result of a piece of text with command 'execute'.
        """
        command = self.append_endline(command)
        params = self.append_endline(params)
        self.service.stdin.write(command.encode('utf8'))
        self.service.stdin.write(params.encode('utf8'))
        self.service.stdin.flush()
        state = self.service.stdout.readline().decode('utf8').strip()
        res = self.service.stdout.readline().decode('utf8').strip()
        return state, res
    
    def append_endline(self, text):
        text += '\n ' if not text.endswith('\n') else ''
        return text


nsp = NerServiceProxy()
# nsp.execute_ner('you can never beat me')
# nsp.open_ner_service(classify=True, pos=False)
# for i in range(2000):
#     print(nsp.execute_ner('you can never beat me '+str(i)+' times'))
#     print(nsp.execute_ner('you can only beat me'+str(30000 - i)+' times'))
#
# nsp.close_ner_service()
# nsp.execute_ner('I hate to be deceived')


def get_ner_service_proxy():
    return nsp


# import subprocess
# p = subprocess.Popen("python my.py", shell=True, close_fds=True,
#                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# strlist = ['shia  sdt\n', 'nimabi\n', 'fuckyou  sdt\n', 'asdasda\n']
# for s in strlist:
#     p.stdin.write(s.encode('utf8'))
#     p.stdin.write('shit\n'.encode('utf8'))
#     p.stdin.flush()
#     print(p.stdout.readline().decode('utf8'))
#
#     if p.poll():
#         print('sub terminated')


class DaemonProcess:
    open = 'open'
    close = 'close'
    exec = 'exec'
    exit = 'exit'
    success = 'success'
    
    def __init__(self, func):
        """
        Instance of this class holds one mp.Process instance using function func,
        and the mp.Process instance holds a subprocess of real ner service wrapped by .
        :param func:
        """
        self.process = None
        self.func = func
        self.inq = mp.Queue()
        self.outq = mp.Queue()
    
    def start(self):
        self.process = mp.Process(target=self.func, args=(self.inq, self.outq))
        self.process.daemon = True
        self.process.start()
    
    def end(self):
        self.close_ner_service()
        self.process.terminate()
    
    def open_ner_service(self, classify, pos):
        self.inq.put(DaemonProcess.open)
        self.inq.put(str(classify))
        self.inq.put(str(pos))
    
    def close_ner_service(self):
        self.inq.put(DaemonProcess.close)
    
    def wait_for_daemon(self):
        if self.outq.get() == DaemonProcess.success:
            return
        else:
            raise ValueError('Service failed.')
    
    def set_text_arr(self, textarr):
        self.inq.put(DaemonProcess.exec)
        self.inq.put(textarr)
    
    def get_text_arr(self):
        return self.outq.get()


class NerServicePool:
    def __init__(self):
        """
        Instance of this class holds multiple instances of DaemonProcess.
        """
        self.dae_pool = None
        self.service_on = False
    
    def start(self, pool_size, classify, pos):
        if self.service_on:
            return
        self.dae_pool = list()
        for i in range(pool_size):
            daeprocess = DaemonProcess(ner_service_daemon)
            daeprocess.start()
            daeprocess.open_ner_service(classify, pos)
            self.dae_pool.append(daeprocess)
        for daeprocess in self.dae_pool:
            daeprocess.wait_for_daemon()
        self.service_on = True
    
    def end(self):
        if not self.service_on:
            return
        for daeprocess in self.dae_pool:
            daeprocess.close_ner_service()
        for daeprocess in self.dae_pool:
            daeprocess.wait_for_daemon()
        self.service_on = False
    
    def execute_ner_multiple(self, textarr):
        blocksize = math.ceil(len(textarr) / len(self.dae_pool))
        res = []
        for idx, dae in enumerate(self.dae_pool):
            arr = textarr[idx * blocksize: (idx + 1) * blocksize]
            dae.set_text_arr(arr)
        for dae in self.dae_pool:
            output = dae.get_text_arr()
            res.extend(output)
        return res
    

def ner_service_daemon(inq, outq):
    ner_service = NerServiceProxy()
    while True:
        command = inq.get()
        if command == DaemonProcess.open:
            classify = bool(inq.get())
            pos = bool(inq.get())
            ner_service.open_ner_service(classify=classify, pos=pos)
            outq.put(DaemonProcess.success)
        if command == DaemonProcess.close:
            ner_service.close_ner_service()
            outq.put(DaemonProcess.success)
        if command == DaemonProcess.exec:
            textarr = inq.get()
            resarr = [ner_service.execute_ner(text) for text in textarr]
            # resarr = []
            # for text in textarr:
            #     resarr.append(ner_service.execute_ner(text))
            # resarr = ner_service.execute_ner_array(textarr)
            outq.put(resarr)
        if command == DaemonProcess.exit:
            return


# dea = DaemonProcess(ner_service_daemon)
# dea.open_ner_service(False, True)
# print(dea.set_text_arr([
#         'you sucks aren\'t you?',
#         'The only limit is your imagination.',
#         'I thought you might like to read the enclosed.',
#         'The radar beam can track a number of targets almost simultaneously.',
#     ]))

# nspool = NerServicePool(3)
# nspool.start()
# nspool.open_ner_service(False, True)
# nspool.execute_ner_multiple([
#         'you sucks aren\'t you?',
#         'The only limit is your imagination.',
#         'I thought you might like to read the enclosed.',
#         'The radar beam can track a number of targets almost simultaneously.',
#         'I\'m creating a multiprocess, which creates a csv file.',
#         'The above part of code takes some time to scrap a webpage',
#         'According to multiprocess daemon documentation by setting',
#         'That occurs before they can start to write so no output will be produced.',
#         'The multiprocessing package offers both local and remote concurrency',
#         'This basic example of data parallelism using Pool',
#         'I\'ve a seed list of URLs from which I need to scrap the data.',
#     ])

nspool = NerServicePool()


def get_ner_service_pool():
    return nspool

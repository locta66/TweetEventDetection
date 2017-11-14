# import __init__
# import numpy as np
# from EventClassifier import EventClassifier
#
# dim=100
# samplenum=20
#
# classifier = EventClassifier(vocab_size=dim, learning_rate=0)
# idfmtx = np.random.random([samplenum,dim])
# print(classifier.predict(idfmtx))
#
# print([classifier.predict([idfmtx[i]])[0] for i in range(samplenum) if classifier.predict([idfmtx[i]])[0]>0.5])
#
#

import multiprocessing as mp

class DC:
    def __init__(self):
        pass

class M:
    def __init__(self, value):
        self.value = value
    
    def printmyself(self, string, dc):
        return str(self.value) + string + str(type(dc))

m = M(233)
dc = DC()

pool = mp.Pool(processes=1)
res_getter = list()
for i in range(3):
    res = pool.apply_async(func=M.printmyself, args=(m, 'shit', dc))
    res_getter.append(res)

pool.close()
pool.join()
results = list()
for i in range(3):
    results.append(res_getter[i].get())

print(results)




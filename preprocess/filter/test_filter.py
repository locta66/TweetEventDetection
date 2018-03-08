from preprocess.filter.filter_effect_check import EffectCheck
from preprocess.filter.filter_utils import readFilesAsJsonList


filter_res = '/home/nfs/cdong/tw/src/models/filter/data/Terrorist.sum'
my_filter = EffectCheck()
my_filter.get_filter_res(filter_res)
data = readFilesAsJsonList(filter_res)
print(len(my_filter.filter(data)))

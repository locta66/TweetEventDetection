from utils.utils_loader import *
from pathlib import Path
import os


def iter_tree():
    in_base_pattern = "/home/nfs/yying/sync/data{}/tweets/"
    out_base_pattern = "/home/nfs/cdong/tw/seeding/Terrorist/queried/negative/full_neg_2012_{}/"
    
    for idx in range(1, 2):
        in_base = in_base_pattern.format(idx)
        out_base = out_base_pattern.format(idx)
        path = Path(out_base)
        if not path.exists():
            path.mkdir()
        
        all_files = fi.listchildren(in_base, fi.TYPE_FILE, concat=True)
        print(len(all_files))
        
        index = 0
        batch_size = 100
        while index * batch_size < len(all_files):
            files = all_files[index * batch_size: (index + 1) * batch_size]
            print(len(files))
            tmu.check_time('iter_tree', print_func=None)
            twarr = au.merge_array([fu.load_twarr_from_bz2(file) for file in files])
            tmu.check_time('iter_tree')
            file_name = "{:0>2}.json".format(index)
            out_file = os.path.join(out_base, file_name)
            print(len(twarr), out_file)
            fu.dump_array(out_file, twarr)
            index += 1


if __name__ == '__main__':
    tmu.check_time()
    iter_tree()
    tmu.check_time()

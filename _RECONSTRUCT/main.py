import __init__
import argparse
from Configure import getconfig
from Summarization import *
import FileIterator as fi


@fu.sync_real_time_counter('_reconstruct main')
def main():
    # fi.iterate_file_tree(getconfig().data_path, summary_files_in_path,
    #                      summary_path='/home/nfs/cdong/tw/testdata/cdong/non')
    # convert_files_multi(getconfig().origin_path, getconfig().summary_path)
    get_tokens_multi(getconfig().origin_path)


def parse_args():
    parser = argparse.ArgumentParser(description="_reconstruct")
    # parser.add_argument('--simp', action='store_true', default=False, help='simplify_files_multi')
    return parser.parse_args()


if __name__ == '__main__':
    main()

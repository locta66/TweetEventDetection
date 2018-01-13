import argparse
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.tweet_keys as tk
import utils.pattern_utils as pu


def main(args):
    base = args.base
    files = args.files
    if not base or not fi.exists(base) or not fi.is_dir(base):
        base = fi.add_sep_if_needed(fi.pwd())
    if not files:
        files = fi.listchildren(base, children_type=fi.TYPE_FILE)
    for file in files:
        real_file = base + file
        if fi.exists(real_file):
            # with open(file, 'r') as fp:
            #     for line in fp.readlines()[0:2]:
            #         print(file, line[50:70])
            twarr = fu.load_array(real_file)
            for tw in twarr:
                print(tw[tk.key_text])
                print('---')
            print('\n{} tws in total'.format(len(twarr)))


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('-b', action='store', dest='base', default='', help='list files to be parsed.')
    parser.add_argument('-f', action='append', dest='files', default=[], help='list files to be parsed.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

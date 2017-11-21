import argparse
from Configure import getconfig
import FileIterator


def main(args):
    import time
    s = time.time()
    
    # FileIterator.set_commands(args.unzip_cmd, args.pos_cmd, args.ner_cmd)
    # if args.unzip:
    #     FileIterator.iterate_file_tree(args.path, FileIterator.unzip_files_in_path)
    if args.summary or args.all_op:
        FileIterator.iterate_file_tree(args.data_path, FileIterator.summary_files_in_path,
                                       summary_path=args.summary_path)
    # if args.pre:
    #     FileIterator.preprocess_summary(args.summary_path)
    print('time elapsed:', time.time() - s, 's')


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess tw streaming data")
    # parser.add_argument('--all_op', action='store_true', default=False,
    #                     help='If perform all the operations below')
    parser.add_argument('--data_path', nargs='?', default=getconfig().data_path,
                        help='Input tweet streaming data path of all months')
    parser.add_argument('--unzip', action='store_true', default=False,
                        help='If unzip files XX.bz2 at the deepest level of file tree')
    parser.add_argument('--summary', action='store_true', default=False,
                        help='If summary filtered tweets into files under summary_path')
    parser.add_argument('--summary_path', nargs='?', default=getconfig().summary_path,
                        help='Filtered tweets organized in days as file XX_XX_XX.txt under this path')
    
    # parser.add_argument('--unzip_cmd', nargs='?', default='bunzip2 -fv ', help='command of unzip')
    # parser.add_argument('--pos_cmd', nargs='?',
    #                     default='???/ark-tweet-nlp-0.3.2/runTagger.sh --input-format json ',
    #                     help='command of pos')
    # parser.add_argument('--ner_cmd', nargs='?',
    #                     default='python /home/nfs/cdong/tw/nlptools/twitter_nlp-master/' +
    #                             'python/ner/extractEntities.py -c -m 512m ',
    #                     help='command of ner')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

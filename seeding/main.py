import __init__
import argparse

from SeedParser import *


def main(args):
    import time
    s = time.time()
    
    seed_parser = SeedParser([
        [{'all_of': ['terror', 'attack', 'death']}, ['2016', '11', '01'], ['2016', '11', '30']],
    ], theme='Terrorist', description='Describes event of terrorist attack', is_seed=True)
    unlb_parser = SeedParser([
        [{'any_of': ['terror', 'attack']}, ['2016', '11', '01'], ['2016', '11', '01']],
    ], theme='Terrorist', description='Event of terrorist attack', is_seed=False)
    
    if args.unlb:
        args.totag = args.untag = False
        parser = unlb_parser
    else:
        parser = seed_parser
    parser.set_base_path(args.seed_path)
    
    if args.query:
        exec_query(args.summary_path, parser)
    if args.totag:
        exec_totag(parser)
    if args.ner:
        exec_ner(parser)
    if args.untag:
        exec_untag(parser)
    
    print('time elapsed:', time.time() - s, 's')


def parse_args():
    parser = argparse.ArgumentParser(description="Seeding information")
    parser.add_argument('--summary_path', nargs='?', default='/home/nfs/cdong/tw/summary/',
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', nargs='?', default='/home/nfs/cdong/tw/seeding/',
                        help='Path for extracted seed instances according to particular query.')
    
    parser.add_argument('--unlb', action='store_true', default=False,
                        help='If query is performed for unlabeled tweets.')
    parser.add_argument('--query', action='store_true', default=False,
                        help='If query tweets from summarized tw files.')
    parser.add_argument('--totag', action='store_true', default=False,
                        help='If makes to tag file from queried tweets.')
    parser.add_argument('--ner', action='store_true', default=False,
                        help='If perform ner on queried file.')
    parser.add_argument('--untag', action='store_true', default=False,
                        help='If updates queried tweets from tagged files.')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

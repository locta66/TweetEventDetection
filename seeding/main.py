import __init__
import argparse
import SeedParser


def main(args):
    import time
    s = time.time()
    
    if args.query:
        SeedParser.parse_querys(args.summary_path, args.seed_path)
    if args.totag:
        SeedParser.create_to_tag(args.seed_path)
    if args.ner:
        SeedParser.perform_ner(args.seed_path)
    
    print('time elapsed:', time.time() - s, 's')


def parse_args():
    parser = argparse.ArgumentParser(description="Seeding data")
    parser.add_argument('--summary_path', nargs='?', default='/home/nfs/cdong/tw/summary/',
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', nargs='?', default='/home/nfs/cdong/tw/seeding/',
                        help='Path for extracted seed instances according to particular query.')
    
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

import __init__
import argparse
import SeedParser


def main(args):
    import time
    s = time.time()
    
    SeedParser.parse_querys(args.summary_path, args.seed_path)
    
    print('time elapsed:', time.time() - s, 's')


def parse_args():
    parser = argparse.ArgumentParser(description="Seeding data")
    parser.add_argument('--summary_path', nargs='?', default='/home/nfs/cdong/tw/summary/',
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path')
    # parser.add_argument('--summary_path', nargs='?', default='/home/nfs/cdong/tw/tempsummary/',
    #                     help='Filtered tweets organized in days as file XX_XX_XX.txt under this path')
    parser.add_argument('--seed_path', nargs='?', default='/home/nfs/cdong/tw/seeding/',
                        help='Path for extracted seed instances according to particular query')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

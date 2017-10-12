import argparse


def main(args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('--seed_path', nargs='?', default='/home/nfs/cdong/tw/seeding/',
                        help='Path for extracted seed instances according to particular query, '
                             'adding trained tensorflow parameters and the corresponding dict.')
    
    parser.add_argument('--unlb', action='store_true', default=False,
                        help='If query is performed for unlabeled tweets.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

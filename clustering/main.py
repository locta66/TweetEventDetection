import argparse
from SeedParser import SeedParser, TestParser
from Main2Clusterer import *

test_parser_1 = TestParser([
    # [{'all_of': ['', '', ], 'any_of':['', '', ], }, ['2016', '', ''], ['2016', '', '']],
    [{'all_of': ['Kabul', 'attack'], 'any_of':['security', 'ministry', 'blast', 'bomb', 'militant', ], },
     ['2016', '9', '5'], ['2016', '9', '10']],
    [{'all_of': ['Mogadishu'], 'any_of':['Shabaab', 'bomb', 'attack', 'toll', 'suicide', ], },
     ['2016', '12', '10'], ['2016', '12', '15']],
    [{'all_of': ['Cairo', 'explosion'], 'any_of':['Shabaab', 'Cathedral', 'Coptic', ], },
     ['2016', '12', '10'], ['2016', '12', '16']],
], theme='Terrorist', description='')


def main(args):
    parser = None
    if args.test:
        parser = test_parser_1
    
    seed_parser = SeedParser(query_list=[], theme='Terrorist', description='')
    for p in [seed_parser, test_parser_1]:
        p.set_base_path(args.seed_path)
    


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('--seed_path', nargs='?', default='/home/nfs/cdong/tw/seeding/',
                        help='Path for queried seed instance, trained parameters and corresponding dict.')
    
    parser.add_argument('--test', action='store_true', default=False,
                        help='If perform actions upon test data.')
    parser.add_argument('--query', action='store_true', default=False,
                        help='If query tweets from summarized tw files.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

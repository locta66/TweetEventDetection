import __init__
import argparse
from SeedParser import SeedParser, TestParser
from Main2Clusterer import *
from Configure import getconfig


seed_parser = SeedParser(query_list=[], theme='Terrorist', description='')
test_parser_1 = TestParser([
    # [{'all_of': ['', '', ], 'any_of':['', '', ], }, ['2016', '', ''], ['2016', '', '']],
    [{'all_of': ['Kabul', ], 'any_of':['University', 'bomb', 'shoot', 'attack', '\Wkill', '\Wtoll', ], },
     ['2016', '8', '23'], ['2016', '8', '28']],
    [{'all_of': ['\WAden\W', ], 'any_of': ['military', '\Wbomb', 'suicide', 'attack', '\Wkill', '\Wtoll', ], },
     ['2016', '8', '28'], ['2016', '9', '3']],
    [{'all_of': ['Kabul', 'attack'], 'any_of':['security', 'ministry', 'blast', '\Wbomb', 'militant', ], },
     ['2016', '9', '5'], ['2016', '9', '10']],
    [{'any_of': ['\Wbomb', '\Wkill', '\Wtoll', 'shopping', 'explosion', 'victim', 'wound', ],
      'all_of': ['Baghdad', ], }, ['2016', '9', '9'], ['2016', '9', '14']],
    [{'all_of': ['Kunduz', ], 'any_of':['gun', 'battle', 'policemen', '\Wkill', 'checkpoint', 'attack', ],
      'none_of':['airstrike', 'year', ], }, ['2016', '10', '3'], ['2016', '10', '8']],
    [{'all_of': ['Baghdad', ], 'any_of':['bomb', 'suicide', 'attack', '\Wkill', '\Wtoll', 'gunmen', ], },
     ['2016', '10', '14'], ['2016', '10', '19']],
    [{'all_of': ['Quetta', ], 'any_of':['police training', 'kill', 'attack', 'hostage', 'militant', ], },
     ['2016', '10', '23'], ['2016', '10', '26']],
    [{'any_of':['Adamawa', '\Wbomb', 'suicide', 'attack', '\Wkill', 'explosion', ],
      'all_of': ['Madagali', ], }, ['2016', '12', '8'], ['2016', '12', '13']],
    [{'all_of': ['Mogadishu'], 'any_of':['Shabaab', '\Wbomb', 'attack', '\Wtoll', 'suicide', ], },
     ['2016', '12', '10'], ['2016', '12', '15']],
    [{'all_of': ['Cairo', 'explosion'], 'any_of':['Shabaab', 'Cathedral', 'Coptic', ], },
     ['2016', '12', '10'], ['2016', '12', '15']],
    [{'all_of': ['\WAden\W', ], 'any_of':['bomb', 'suicide', 'attack', '\Wkill', '\Wtoll', 'soldier', ], },
     ['2016', '12', '17'], ['2016', '12', '22']],
    [{'all_of': ['Karak', ], 'any_of':['bomb', 'suicide', 'attack', '\Wkill', '\Wtoll', 'security', ], },
     ['2016', '12', '17'], ['2016', '12', '22']],
], theme='Terrorist', description='', outterid='1')


def main(args):
    parser = test_parser_1
    # if args.test:
    #     parser = test_parser_1
    
    for p in [seed_parser, test_parser_1]:
        p.set_base_path(args.seed_path)
    if args.query:
        exec_query(args.summary_path, parser)
    if args.pred:
        exec_classification(seed_parser, test_parser_1)
    if args.ner:
        exec_ner(parser)
    if args.temp:
        exec_temp(parser)


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('--summary_path', nargs='?', default=getconfig().summary_path,
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', nargs='?', default=getconfig().seed_path,
                        help='Path for queried seed instance, trained parameters and corresponding dict.')
    
    parser.add_argument('--test', action='store_true', default=False,
                        help='If perform actions upon test data.')
    parser.add_argument('--query', action='store_true', default=False,
                        help='If query tweets from summarized tw files.')
    parser.add_argument('--ner', action='store_true', default=False,
                        help='If perform ner on queried file.')
    parser.add_argument('--pred', action='store_true', default=False,
                        help='If perform prediction on the test data.')
    # parser.add_argument('--cluster', action='store_true', default=False,
    #                     help='If perform clustering on some tweet stream.')
    parser.add_argument('--temp', action='store_true', default=False,
                        help='Just a temp function.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

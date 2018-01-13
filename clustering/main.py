import argparse
from seeding.seed_parser import SeedParser, TestParser
from clustering.main2clusterer import *
from config.configure import getcfg


seed_parser = SeedParser(query_list=[], theme='Terrorist', description='')
test_parser_1 = TestParser([
    # [{'all_of': ['', '', ], 'any_of':['', '', ], }, ['2016', '', ''], ['2016', '', '']],
    
    # [{'all_of': ['Kabul', ], 'any_of':['University', 'bomb', 'shoot', 'attack', '\Wkill', '\Wtoll', ], },
    #  ['2016', '8', '23'], ['2016', '8', '28']],
    # [{'all_of': ['\WAden\W', ], 'any_of': ['military', '\Wbomb', 'suicide', 'attack', '\Wkill', '\Wtoll', ], },
    #  ['2016', '8', '28'], ['2016', '9', '3']],
    # [{'all_of': ['Kabul', 'attack'], 'any_of':['security', 'ministry', 'blast', '\Wbomb', 'militant', ], },
    #  ['2016', '9', '5'], ['2016', '9', '10']],
    # [{'any_of': ['\Wbomb', '\Wkill', '\Wtoll', 'shopping', 'explosion', 'victim', 'wound', ],
    #   'all_of': ['Baghdad', ], }, ['2016', '9', '9'], ['2016', '9', '14']],
    # [{'all_of': ['Kunduz', ], 'any_of':['gun', 'battle', 'policemen', '\Wkill', 'checkpoint', 'attack', ],
    #   'none_of':['airstrike', 'year', ], }, ['2016', '10', '3'], ['2016', '10', '8']],
    # [{'all_of': ['Baghdad', ], 'any_of':['bomb', 'suicide', 'attack', '\Wkill', '\Wtoll', 'gunmen', ], },
    #  ['2016', '10', '14'], ['2016', '10', '19']],
    # [{'all_of': ['Quetta', ], 'any_of':['police training', 'kill', 'attack', 'hostage', 'militant', ], },
    #  ['2016', '10', '23'], ['2016', '10', '26']],
    # [{'any_of':['Adamawa', '\Wbomb', 'suicide', 'attack', '\Wkill', 'explosion', ],
    #   'all_of': ['Madagali', ], }, ['2016', '12', '8'], ['2016', '12', '13']],
    # [{'all_of': ['Mogadishu'], 'any_of':['Shabaab', '\Wbomb', 'attack', '\Wtoll', 'suicide', ], },
    #  ['2016', '12', '10'], ['2016', '12', '15']],
    # [{'all_of': ['Cairo', 'explosion'], 'any_of':['Shabaab', 'Cathedral', 'Coptic', ], },
    #  ['2016', '12', '10'], ['2016', '12', '15']],
    # [{'all_of': ['\WAden\W', ], 'any_of':['bomb', 'suicide', 'attack', '\Wkill', '\Wtoll', 'soldier', ], },
    #  ['2016', '12', '17'], ['2016', '12', '22']],
    # [{'all_of': ['Karak', ], 'any_of':['bomb', 'suicide', 'attack', '\Wkill', '\Wtoll', 'security', ], },
    #  ['2016', '12', '17'], ['2016', '12', '22']],
    #
    # [{'all_of': ['Indonesia', 'earthquake', ], 'any_of':['hit', 'magnitude', 'strike', ], },
    #  ['2016', '12', '6'], ['2016', '12', '10']],
    # [{'all_of': ['India', 'earthquake', ], 'any_of':['hit', 'magnitude', 'strike', 'intensity', ], },
    #  ['2016', '1', '2'], ['2016', '1', '7']],
    # [{'any_of': ['Romania', 'Belgium', 'France', 'Seine\W', 'Germany', 'disaster',
    #              '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
    #   'all_of': ['flood', ], }, ['2016', '5', '27'], ['2016', '6', '7']],
    # [{'any_of': ['catastrophic', '\Wdead\W', 'death', 'FEMA', 'flooding', 'rainfall', 'state of emergency', ],
    #   'all_of': ['Louisiana', 'flood', ], }, ['2016', '8', '10'], ['2016', '8', '17']],
    # [{'any_of': ['wildfire', 'Great Smoky Mountain', 'damage', 'destroy', 'burn', '\Wacre\W', 'evacuat', ],
    #   'all_of': ['National Park', ], }, ['2016', '11', '23'], ['2016', '11', '30']],
    # [{'any_of': ['\Wfire', 'wildfire', 'damage', 'burned', '\Wburn\W', '\Wacre\W', 'evacuat', ],
    #   'all_of': ['Hayden', ], }, ['2016', '7', '7'], ['2016', '7', '15']],
    # [{'all_of': ['earthquake', 'Ecuador'], 'any_of':['hit', 'magnitude', 'strike', 'toll', 'Esmeraldas'], },
    #  ['2016', '4', '15'], ['2016', '4', '19']],
    # [{'all_of': ['earthquake', 'Italy', ], 'any_of':['hit', 'magnitude', 'strike', 'toll', ], },
    #  ['2016', '8', '24'], ['2016', '8', '29']],
    #
    # [{'any_of': ['gunman', 'hostage', 'shoot', '\Wshot', ],
    #   'all_of': ['Orlando', 'nightclub', 'attack', ], }, ['2016', '6', '11'], ['2016', '6', '16']],
    # [{'any_of': ['attack', '\Wbomb', 'explode', 'hostage', '\Wkill', 'injured', ],
    #   'all_of': ['Mogadishu', ], }, ['2016', '6', '24'], ['2016', '6', '29']],
    # [{'any_of': ['attack', '\Wbomb', 'headquarter', 'security', '\Wkill'],
    #   'all_of': ['\WAden\W', 'airport'], }, ['2016', '7', '5'], ['2016', '7', '11']],
    # [{'all_of': ['Nice', 'truck'], 'any_of': ['Bastille', 'crowd', 'gunfire', ], },
    #  ['2016', '7', '13'], ['2016', '7', '18']],
    # [{'all_of': ['Mogadishu', ], 'any_of': ['\Wkill', '\Wbomb', 'suicide', 'Shabaab', ], },
    #  ['2016', '7', '25'], ['2016', '7', '31']],
    # [{'all_of': ['Kokrajhar', ], 'any_of': ['Assam', 'Bodo', '\Wgun\W', 'grenade', '\Wkill', 'shoot', ], },
    #  ['2016', '8', '4'], ['2016', '8', '9']],
    # [{'all_of': ['Quetta', ], 'any_of': ['attack', '\Wbomb', 'hospital', '\Wkill', 'suicide', ], },
    #  ['2016', '8', '7'], ['2016', '8', '10']],
    # [{'all_of': ['PKK', 'Turkey', ], 'any_of': ['attack', '\Wbomb', 'soldier', ]},
    #  ['2016', '8', '9'], ['2016', '8', '14']],

], theme='Terrorist', description='', outterid='1')


def main(args):
    parser = test_parser_1
    for p in [seed_parser, test_parser_1]:
        p.set_base_path(getcfg().seed_path)
    
    # if args.query:
    #     exec_query(args.summary_path, parser)
    # if args.pred:
    #     exec_classification(seed_parser, test_parser_1)
    if args.ner:
        exec_ner(parser)
    if args.clu:
        exec_cluster(parser)
    if args.temp:
        exec_temp(parser)
    if args.anly:
        exec_analyze(parser)


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('--summary_path', nargs='?', default=getcfg().summary_path,
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', nargs='?', default=getcfg().seed_path,
                        help='Path for queried seed instance, trained parameters and corresponding dict.')
    
    parser.add_argument('--test', action='store_true', default=False, help='If perform actions upon test data.')
    parser.add_argument('--query', action='store_true', default=False, help='If query tweets from summarized tw files.')
    parser.add_argument('--ner', action='store_true', default=False, help='If perform ner on queried file.')
    parser.add_argument('--pred', action='store_true', default=False, help='If perform prediction on the test data.')
    parser.add_argument('--clu', action='store_true', default=False, help='If perform clustering on some tweet stream.')
    parser.add_argument('--temp', action='store_true', default=False, help='Just a temp function, maybe for making data.')
    parser.add_argument('--anly', action='store_true', default=False, help='analyze the clustering results.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

import __init__
import argparse

from SeedParser import *
from Main2Parser import *
from Configure import getconfig

seed_parser = SeedParser([
    # [{'any_of': ['', '', ], 'all_of': ['', '', ], }, ['2016', '', ''], ['2016', '', '']],
    [{'all_of': ['Paris', 'police', 'station', ], 'any_of': ['attack', 'knife', ], },
     ['2016', '1', '6'], ['2016', '1', '10']],
    [{'all_of': ['Quetta', ], 'any_of': ['\Wbomb', 'security ', 'suicide', 'polio', 'policemen', ], },
     ['2016', '1', '12'], ['2016', '1', '17']],
    [{'all_of': ['\WAdde\W', ], 'any_of': ['Union', 'attack', 'massive', 'siege', ], },
     ['2016', '1', '14'], ['2016', '1', '19']],
    [{'all_of': ['Dalori', 'Boko', ], 'any_of': ['attack', '\Wkill', ], },
     ['2016', '1', '29'], ['2016', '2', '3']],
    [{'all_of': ['Ankara', 'explosion', ], 'any_of': ['attack', '\Wbomb', 'military', ], },
     ['2016', '2', '16'], ['2016', '2', '21']],
    [{'all_of': ['Cameroon\W', ], 'any_of': ['\Wbomb', 'blast', '\Wkill', 'suicide', ], },
     ['2016', '2', '18'], ['2016', '2', '23']],
    [{'any_of': ['attack', '\Wbomb', '\Wkill', 'suicide', ],
      'all_of': ['Homs', ], 'none_of': ['airstrike', ]}, ['2016', '2', '20'], ['2016', '2', '25']],
    [{'any_of': ['Grand', 'AQIM', 'attack', 'gunmen', '\Wkill', '\Wshot', 'shoot', ],
      'all_of': ['Bassam', ], }, ['2016', '3', '12'], ['2016', '3', '17']],
    [{'all_of': ['Brussels', 'attack', ], 'any_of': ['\Wbomb', '\Wkill', 'suicide', ], },
     ['2016', '3', '22'], ['2016', '3', '26']],
    [{'all_of': ['Chhattisgarh', 'CRPF', ], 'any_of': ['attack', 'Dantewada', ], },
     ['2016', '3', '29'], ['2016', '4', '3']],
    [{'any_of': ['\WAbu\W', 'Sayyaf', 'clash', '\Wgun\W', 'militant', ],
      'all_of': ['Philippine', 'soldier', ], }, ['2016', '4', '8'], ['2016', '4', '13']],
    [{'all_of': ['Kabul', 'attack', ], 'any_of': ['Afghanistan', '\Wbomb', '\Wgun\W', 'security', ], },
     ['2016', '4', '18'], ['2016', '4', '23']],
    [{'any_of': ['murder', 'raid', 'injure', 'suicide', '\Wbomb', 'explosion', '\Wkill', 'wound',
                 'troop', 'Boko Haram', 'terrorist'],
      'all_of': ['Borno', ], }, ['2016', '4', '23'], ['2016', '4', '29']],
    [{'all_of': ['Somalia', ], 'any_of': ['attack', '\Wbomb', 'military', 'recapture', 'Shabaab', ], },
     ['2016', '4', '30'], ['2016', '5', '5']],
    [{'all_of': ['Munich', 'attack', ], 'any_of': ['knife', 'railway', ], },
     ['2016', '5', '9'], ['2016', '5', '14']],
    [{'all_of': ['Jableh', ], 'any_of': ['attack', '\Wbomb', 'military', 'suicide', ], },
     ['2016', '5', '22'], ['2016', '5', '27']],
    [{'all_of': ['Boko', 'Bosso', ], },
     ['2016', '6', '3'], ['2016', '6', '9']],
    [{'any_of': ['gunman', 'hostage', 'shoot', '\Wshot', ],
      'all_of': ['Orlando', 'nightclub', 'attack', ], }, ['2016', '6', '11'], ['2016', '6', '16']],
    [{'any_of': ['attack', '\Wbomb', 'explode', 'hostage', '\Wkill', 'injured', ],
      'all_of': ['Mogadishu', ], }, ['2016', '6', '24'], ['2016', '6', '29']],
    [{'any_of': ['attack', '\Wbomb', 'headquarter', 'security', '\Wkill'],
      'all_of': ['\WAden\W', 'airport'], }, ['2016', '7', '5'], ['2016', '7', '11']],
    [{'all_of': ['Nice', 'truck'], 'any_of': ['Bastille', 'crowd', 'gunfire', ], },
     ['2016', '7', '13'], ['2016', '7', '18']],
    [{'all_of': ['Mogadishu', ], 'any_of': ['\Wkill', '\Wbomb', 'suicide', 'Shabaab', ], },
     ['2016', '7', '25'], ['2016', '7', '31']],
    [{'all_of': ['Kokrajhar', ], 'any_of': ['Assam', 'Bodo', '\Wgun\W', 'grenade', '\Wkill', 'shoot', ], },
     ['2016', '8', '4'], ['2016', '8', '9']],
    [{'all_of': ['Quetta', ], 'any_of': ['attack', '\Wbomb', 'hospital', '\Wkill', 'suicide', ], },
     ['2016', '8', '7'], ['2016', '8', '10']],
    [{'all_of': ['PKK', 'Turkey', ], 'any_of': ['attack', '\Wbomb', 'soldier', ]},
     ['2016', '8', '9'], ['2016', '8', '14']],
], theme='Terrorist', description='Describes event of terrorist attack')

unlb_parser = UnlbParser([
    [{'all_of': ['attack', ],
      'any_of': ['terror', 'attack', 'fight', 'assault', 'death', '\Wgun\W', '\Wfire\W', '\Wbomb',
                 'battle', '\Wkill', 'explode', 'explosion', 'wound', 'injure', 'deadly', 'shoot', ],
      }, ['2016', '8', '15'], ['2016', '10', '31']],
], theme='Terrorist', description='Unidentified event of terrorist attack')

cntr_parser = CounterParser([
    [{'any_of': ['happy', 'glad', 'nice', 'party', 'good', 'excellent', 'wonderful', 'magnificent'],
      'none_of': ['terror', 'attack', 'fight', 'assault', 'death', '\Wgun\W', 'fire', 'bomb',
                  'battle', 'kill', 'explode', 'explosion', 'wound', 'injure', 'deadly', 'shoot', ]
      }, ['2016', '4', '1'], ['2016', '8', '13']],
    [{'any_of': ['tornado', 'struck', 'cyclone', 'extratropical', 'state of emergency', 'record-breaking',
                 'damage', 'destroy', 'supercell', ],
      'all_of': ['Manzanita', ], }, ['2016', '10', '13'], ['2016', '10', '18']],
    [{'any_of': ['wildfire', 'Great Smoky Mountain', 'damage', 'destroy', 'burn', '\Wacre\W', 'evacuat', ],
      'all_of': ['National Park', ], }, ['2016', '11', '23'], ['2016', '11', '30']],
    [{'any_of': ['\Wfire', 'wildfire', 'damage', 'burned', '\Wburn\W', '\Wacre\W', 'evacuat', ],
      'all_of': ['Hayden', ], }, ['2016', '7', '7'], ['2016', '7', '15']],
    [{'all_of': ['earthquake', 'Italy', ], 'any_of': ['hit', 'magnitude', 'strike', 'toll', ], },
     ['2016', '8', '24'], ['2016', '8', '28']],
    [{'all_of': ['earthquake', 'Zealand', ], 'any_of': ['hit', 'magnitude', ], },
     ['2016', '11', '14'], ['2016', '11', '18']],
    [{'all_of': ['earthquake', 'Indonesia', ], 'any_of': ['hit', 'magnitude', 'strike', ], },
     ['2016', '12', '6'], ['2016', '12', '10']],
    [{'all_of': ['earthquake', 'India', ], 'any_of': ['hit', 'magnitude', 'strike', 'intensity', ], },
     ['2016', '1', '2'], ['2016', '1', '7']],
    [{'any_of': ['Romania', 'Belgium', 'France', 'Seine\W', 'Germany', 'disaster',
                 '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['flood', ], }, ['2016', '5', '27'], ['2016', '6', '7']],
    [{'any_of': ['catastrophic ', 'rainfall', 'state of emergency', 'death', 'flooding', '\Wdead\W',
                 'FEMA', ],
      'all_of': ['Louisiana', 'flood', ], }, ['2016', '8', '10'], ['2016', '8', '17']],
    [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'kill', 'missing', 'rainfall', 'rescue', ],
      'all_of': ['Pakistan', 'flood', ], }, ['2016', '4', '2'], ['2016', '4', '8']],
], theme='Terrorist', description='Not Event of terrorist attack')

# seed_parser = SeedParser([
#     # [{'any_of': ['', '', ], 'all_of': ['', '', ], }, ['2016', '', ''], ['2016', '', '']],
#     [{'any_of': ['tornado', 'struck', 'cyclone', 'extratropical', 'state of emergency', 'record-breaking',
#                  'damage', 'destroy', 'supercell', ],
#       'all_of': ['Manzanita', ], }, ['2016', '10', '13'], ['2016', '10', '18']],
#
#     [{'any_of': ['wildfire', 'Great Smoky Mountain', 'damage', 'destroy', 'burn', '\Wacre\W', 'evacuat', ],
#       'all_of': ['National Park', ], }, ['2016', '11', '23'], ['2016', '11', '30']],
#     [{'any_of': ['\Wfire', 'wildfire', 'damage', 'burned', '\Wburn\W', '\Wacre\W', 'evacuat', ],
#       'all_of': ['Hayden', ], }, ['2016', '7', '7'], ['2016', '7', '15']],
#
#     [{'all_of': ['earthquake', 'Ecuador'], 'any_of':['hit', 'magnitude', 'strike', 'toll', 'Esmeraldas'], },
#      ['2016', '4', '15'], ['2016', '4', '19']],
#     [{'all_of': ['earthquake', 'Italy', ], 'any_of':['hit', 'magnitude', 'strike', 'toll', ], },
#      ['2016', '8', '24'], ['2016', '8', '28']],
#     [{'all_of': ['earthquake', 'Zealand', ], 'any_of':['hit', 'magnitude', ], },
#      ['2016', '11', '14'], ['2016', '11', '18']],
#     [{'all_of': ['earthquake', 'Tanzania'], 'any_of':['hit', 'magnitude', 'strike', ], },
#      ['2016', '9', '10'], ['2016', '9', '14']],
#     [{'all_of': ['earthquake', 'Indonesia', ], 'any_of':['hit', 'magnitude', 'strike', ], },
#      ['2016', '12', '6'], ['2016', '12', '10']],
#     [{'all_of': ['earthquake', 'India', ], 'any_of':['hit', 'magnitude', 'strike', 'intensity', ], },
#      ['2016', '1', '2'], ['2016', '1', '7']],
#
#     [{'any_of': ['Romania', 'Belgium', 'France', 'Seine\W', 'Germany', 'disaster',
#                  '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
#       'all_of': ['flood', ], }, ['2016', '5', '27'], ['2016', '6', '7']],
#     [{'any_of': ['catastrophic', '\Wdead\W', 'death', 'FEMA', 'flooding', 'rainfall', 'state of emergency', ],
#       'all_of': ['Louisiana', 'flood', ], }, ['2016', '8', '10'], ['2016', '8', '17']],
#     [{'any_of': ['disaster', '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
#       'all_of': ['Virginia', 'flood', ], }, ['2016', '6', '22'], ['2016', '6', '28']],
#     [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'missing', 'rainfall', 'rescue', ],
#       'all_of': ['Maryland', 'flood', ], }, ['2016', '7', '29'], ['2016', '8', '7']],
#     [{'any_of': ['damage', 'disaster', '\Wdead\W', 'death', 'kill', 'missing', 'rainfall', 'rescue', ],
#       'all_of': ['Pakistan', 'flood', ], }, ['2016', '4', '2'], ['2016', '4', '8']],
# ], theme='NaturalDisaster', description='Describes event of natural disaster events')


@sync_real_time_counter('main')
def main(args):
    if args.unlb:
        args.ner = False
        query_func = exec_query_unlabelled
        parser = unlb_parser
    elif args.cntr:
        # args.totag = args.untag = False
        query_func = exec_query_counter
        parser = cntr_parser
    else:
        # args.totag = args.untag = False
        query_func = exec_query
        parser = seed_parser
    for p in [seed_parser, unlb_parser, cntr_parser]:
        p.set_base_path(args.seed_path)
    
    if args.query:
        query_func(args.summary_path, parser)
    if args.ner:
        exec_ner(parser)
    if args.train:
        exec_train(seed_parser, unlb_parser, cntr_parser)
    if args.temp:
        temp(cntr_parser)
    
    if args.pre_test:
        exec_pre_test(args.test_data_path)
    if args.train_outer:
        exec_train_with_outer(seed_parser, unlb_parser, cntr_parser)


def parse_args():
    parser = argparse.ArgumentParser(description="Seeding information")
    parser.add_argument('--summary_path', default=getconfig().summary_path,
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', default=getconfig().seed_path,
                        help='Path for extracted seed instances according to particular query.')
    
    parser.add_argument('--unlb', action='store_true', default=False,
                        help='If query is performed for unlabeled tweets.')
    parser.add_argument('--cntr', action='store_true', default=False,
                        help='If query is performed for counter tweets.')
    
    parser.add_argument('--query', action='store_true', default=False,
                        help='If query tweets from summarized tw files.')
    parser.add_argument('--ner', action='store_true', default=False,
                        help='If perform ner on queried file.')
    # parser.add_argument('--totag', action='store_true', default=False,
    #                     help='If makes to tag file from queried tweets.')
    # parser.add_argument('--untag', action='store_true', default=False,
    #                     help='If updates queried tweets from tagged files.')
    parser.add_argument('--train', action='store_true', default=False,
                        help='If train the model according to the queried tweets, with internal logic.')
    parser.add_argument('--temp', action='store_true', default=False,
                        help='Just a temp function.')
    
    parser.add_argument('--test_data_path', default=getconfig().test_data_path,
                        help='Path for test data from dzs.')
    parser.add_argument('--pre_test', action='store_true', default=False,
                        help='Just a temp function to preprocess data from dzs.')
    parser.add_argument('--train_outer', action='store_true', default=False,
                        help='If train and test the model with data from dzs.')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

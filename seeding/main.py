import __init__
import argparse

from SeedParser import *
from Main2Parser import *


seed_parser = SeedParser([
    # [{'all_of': ['', '', ], 'any_of':['', '', ], }, ['2016', '', ''], ['2016', '', '']],
    
    [{'all_of': ['Paris', 'police', 'station', ], 'any_of':['attack', 'knife', ], },
     ['2016', '1', '6'], ['2016', '1', '10']],
    [{'all_of': ['\WAdde\W', ], 'any_of':['Union', 'attack', 'siege', 'massive', ], },
     ['2016', '1', '14'], ['2016', '1', '18']],
    [{'all_of': ['Dalori', 'Boko', ], 'any_of':['attack', 'kill', ], },
     ['2016', '1', '29'], ['2016', '2', '3']],
    [{'all_of': ['Ankara', 'explosion', ], 'any_of':['bomb', 'military', 'attack', ], },
     ['2016', '2', '16'], ['2016', '2', '21']],
    [{'all_of': ['Chhattisgarh', 'CRPF', ], 'any_of':['attack', 'Dantewada', ], },
     ['2016', '3', '29'], ['2016', '4', '3']],
    [{'all_of': ['Philippine', 'soldier', ], 'any_of':['\WAbu\W', 'Sayyaf', '\Wgun\W', 'militant', 'clash', ], },
     ['2016', '4', '8'], ['2016', '4', '13']],
    [{'all_of': ['Kabul', 'attack', ], 'any_of': ['Afghanistan', 'security', 'bomb', 'gun', ], },
     ['2016', '4', '18'], ['2016', '4', '23']],
    [{'all_of': ['Somalia', ], 'any_of':['bomb', 'military', 'attack', 'recapture', 'Shabaab', ], },
     ['2016', '4', '30'], ['2016', '5', '5']],
    [{'all_of': ['Munich', 'attack', ], 'any_of':['knife', 'railway', ], },
     ['2016', '5', '9'], ['2016', '5', '14']],
    [{'all_of': ['Boko', 'Bosso', ], },
     ['2016', '6', '3'], ['2016', '6', '9']],
    [{'all_of': ['Orlando', 'nightclub', 'attack', ], 'any_of':['shoot', 'shot', 'gunman', 'hostage', ], },
     ['2016', '6', '11'], ['2016', '6', '16']],
    [{'all_of': ['Aden', 'airport'], 'any_of': ['headquarter', 'attack', 'bomb', 'security', ], },
     ['2016', '7', '5'], ['2016', '7', '11']],
    [{'all_of': ['Nice', 'truck'], 'any_of':['gunfire', 'Bastille', 'crowd', ], },
     ['2016', '7', '13'], ['2016', '7', '18']],
    [{'all_of': ['PKK', 'Turkey', ], 'any_of':['bomb', 'soldier', 'attack']},
     ['2016', '8', '9'], ['2016', '8', '14']],
    [{'all_of': ['Kabul', 'attack'], 'any_of':['security', 'ministry', 'blast', 'bomb', 'militant', ], },
     ['2016', '9', '5'], ['2016', '9', '10']],
    [{'all_of': ['Mogadishu'], 'any_of':['Shabaab', 'bomb', 'attack', 'toll', 'suicide', ], },
     ['2016', '12', '10'], ['2016', '12', '15']],
    [{'all_of': ['Cairo', 'explosion'], 'any_of':['Shabaab', 'Cathedral', 'Coptic', ], },
     ['2016', '12', '10'], ['2016', '12', '16']],
], theme='Terrorist', description='Describes event of terrorist attack')

unlb_parser = UnlbParser([
    [{
      'all_of': ['terror', 'attack', ],
      'any_of': ['death', 'assault', 'bomb', 'explode', 'explosion', 'knife', 'suicide', 'murder',
                 'flee', 'panic', '\Wcry\W', 'kill', '\Wgun\W', 'shoot', 'fire', 'soldier', ]
     }, ['2016', '10', '1'], ['2016', '10', '31']],
], theme='Terrorist', description='Unidentified event of terrorist attack')

cntr_parser = CounterParser([
    [{
      'any_of': ['happy', 'glad', 'nice', 'party', 'good', 'excellent', 'wonderful', 'magnificent'],
      'none_of': ['terror', 'attack', 'fight', 'assault', 'death', '\Wgun\W', 'fire', 'death', 'battle']},
     ['2016', '10', '1'], ['2016', '11', '30']],
], theme='Terrorist', description='Not Event of terrorist attack')

# seed_parser = SeedParser([
#     # [{'all_of': ['', '']}, ['2016', '', ''], ['2016', '', '']],
#
#     [{'all_of': ['', '']}, ['2016', '', ''], ['2016', '', '']],
#
#     # [{'all_of': ['earthquake', 'Zealand', ], 'any_of':['hit', 'magnitude', ], },
#     #  ['2016', '11', '14'], ['2016', '11', '18']],
#     # [{'all_of': ['earthquake', 'Tanzania'], 'any_of':['hit', 'magnitude', 'strike', ], },
#     #  ['2016', '09', '10'], ['2016', '09', '14']],
#     # [{'all_of': ['earthquake', 'Indonesia', ], 'any_of':['hit', 'magnitude', 'strike', ], },
#     #  ['2016', '12', '06'], ['2016', '12', '10']],
#     # [{'all_of': ['earthquake', 'Italy', ], 'any_of':['hit', 'magnitude', 'strike', 'toll', ], },
#     #  ['2016', '08', '24'], ['2016', '08', '28']],
#     # [{'all_of': ['earthquake', 'Ecuador'], 'any_of':['hit', 'magnitude', 'strike', 'toll', 'Esmeraldas'], },
#     #  ['2016', '', ''], ['2016', '', '']],
# ], theme='Earthquake', description='Describes event of earthquake events')


def main(args):
    import time
    s = time.time()
    
    query_func = exec_query
    if args.unlb:
        args.totag = args.untag = False
        query_func = exec_query_unlabelled
        parser = unlb_parser
    elif args.cntr:
        args.totag = args.untag = False
        parser = cntr_parser
    else:
        parser = seed_parser
    seed_parser.set_base_path(args.seed_path)
    unlb_parser.set_base_path(args.seed_path)
    cntr_parser.set_base_path(args.seed_path)
    
    if args.query:
        query_func(args.summary_path, parser)
        print('Query time elapsed:', time.time() - s, 's')
        s = time.time()
    if args.totag:
        exec_totag(parser)
    if args.untag:
        exec_untag(parser)
    if args.ner:
        exec_ner(parser)
    if args.train:
        exec_train(seed_parser, unlb_parser, cntr_parser)
    # if args.temp:
    #     temp(cntr_parser)
    
    print('time elapsed:', time.time() - s, 's')


def parse_args():
    parser = argparse.ArgumentParser(description="Seeding information")
    parser.add_argument('--summary_path', nargs='?', default='/home/nfs/cdong/tw/summary/',
                        help='Filtered tweets organized in days as file XX_XX_XX_XX.sum under this path.')
    parser.add_argument('--seed_path', nargs='?', default='/home/nfs/cdong/tw/seeding/',
                        help='Path for extracted seed instances according to particular query.')
    
    parser.add_argument('--unlb', action='store_true', default=False,
                        help='If query is performed for unlabeled tweets.')
    parser.add_argument('--cntr', action='store_true', default=False,
                        help='If query is performed for counter tweets.')
    
    parser.add_argument('--query', action='store_true', default=False,
                        help='If query tweets from summarized tw files.')
    parser.add_argument('--totag', action='store_true', default=False,
                        help='If makes to tag file from queried tweets.')
    parser.add_argument('--ner', action='store_true', default=False,
                        help='If perform ner on queried file.')
    parser.add_argument('--untag', action='store_true', default=False,
                        help='If updates queried tweets from tagged files.')
    parser.add_argument('--train', action='store_true', default=False,
                        help='If train the model according to the queried tweets, with internal logic.')
    # parser.add_argument('--temp', action='store_true', default=False,
    #                     help='.')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

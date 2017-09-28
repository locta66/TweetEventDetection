import os
import re
import random
import shutil

import FileIterator
from JsonParser import JsonParser
from SeedQuery import SeedQuery
from NerServiceProxy import get_ner_service_pool
from WordFreqCounter import WordFreqCounter
from EventClassifier import EventClassifier

import numpy as np


class SeedParser(JsonParser):
    def __init__(self, query_list, theme):
        JsonParser.__init__(self)
        self.theme = theme
        self.query_list = query_list
        self.seed_query_list = [SeedQuery(*query) for query in self.query_list]
        self.added_twarr = list()
        self.not_added_twarr = list()
        self.localcounter = WordFreqCounter()
        self.globalcounter = WordFreqCounter()
    
    def read_tweet_from_json_file(self, file, filtering=False):
        if not self.is_file_of_query_date(file):
            return
        with open(file) as json_file:
            for line in json_file.readlines():
                tw = self.parse_text(line, filtering=False)
                tw_af = self.attribute_filter(tw, self.tweet_desired_attrs)
                if 'user' in tw_af:
                    tw_af['user'] = self.attribute_filter(tw_af['user'], self.user_desired_attrs)
                tw_added = False
                for seed_query in self.seed_query_list:
                    tw_added = seed_query.append_desired_tweet(tw_af, usingtwtime=False) or tw_added
                if tw_added:
                    self.added_twarr.append(tw_af)
                else:
                    if random.random() < 1 / 400:
                        self.not_added_twarr.append(tw)
    
    def twarr_ner(self, twarr):
        ner_text_arr = get_ner_service_pool().execute_ner_multiple([tw['text'] for tw in twarr])
        if not len(ner_text_arr) == len(twarr):
            raise ValueError("Line number inconsistent, error occurs during NER")
        for idx, ner_text in enumerate(ner_text_arr):
            wordlabels = self.parse_ner_text_into_wordlabels(ner_text)
            wordlabels = self.remove_noneword_from_wordlabels(wordlabels)
            twarr[idx]['pos'] = wordlabels
        return twarr
    
    def sort_idf_of_added_tweet(self, localidfthreashold=0.0, globalidfthreashold=0.0):
        localcounter = self.localcounter
        globalcounter = self.globalcounter
        print('added docs', len(self.added_twarr))
        for tw in self.twarr_ner(self.added_twarr):
            localcounter.expand_dict_and_count_df_from_wordlabels(tw['pos'])
        print('not added docs', len(self.not_added_twarr))
        for tw in self.twarr_ner(self.not_added_twarr):
            globalcounter.expand_dict_and_count_df_from_wordlabels(tw['pos'])
        
        print('pre vocabulary', localcounter.vocabulary_size())
        localcounter.reserve_word_by_idf_threshold(rsv_cond=lambda idf: idf < localidfthreashold)
        print('mid vocabulary', localcounter.vocabulary_size())
        globalcounter.reserve_word_by_idf_threshold(rsv_cond=lambda idf: idf < globalidfthreashold)
        local_global_common = set(localcounter.vocabulary()).intersection(set(globalcounter.vocabulary()))
        localcounter.remove_words(list(local_global_common), updateid=True)
        print('post vocabulary', localcounter.vocabulary_size())
        print('common word', local_global_common)
        
        # print('\n')
        # for i in range(400):
        #     print(self.added_twarr[i]['text'])
        #     wordlabels = self.added_twarr[i]['pos']
        #     v = localcounter.idf_vector_of_wordlabels(wordlabels)
        #     for word in [wordlabel[0].lower() for wordlabel in wordlabels]:
        #         if not localcounter.is_word_in_dict(word):
        #             continue
        #         wordid = localcounter.word_id(word)
        #         print(word, '\t\t', localcounter.twdict.worddict[word]['idf'], '\t\t', v[wordid])
        #     print('\n')
        
        print('start train')
        classifier = EventClassifier(vocab_size=localcounter.vocabulary_size(), learning_rate=2e-2,
                                     unlbreg_lambda=0.2, l2reg_lambda=0.1)
        print('train len', int(len(self.added_twarr) * 8 / 10))
        train_twarr = self.added_twarr[int(len(self.added_twarr) * 8 / 10):]
        train_idf_mtx = []
        for tw in train_twarr:
            idfvec, added = localcounter.idf_vector_of_wordlabels(tw['pos'])
            train_idf_mtx.append(idfvec * np.log(len(added) + 2))
        import time
        s = time.time()
        classifier.train_steps(500, 1e-5, None, train_idf_mtx, unlby=[[0.2]])
        print('training time ', time.time()-s, 's')
        
        print(classifier.get_value(classifier.thetaEb))
        # print(classifier.get_value(classifier.thetaEW))
        
        test_twarr = self.added_twarr[0: int(len(self.added_twarr) * 95 / 100)]
        test_idf_mtx = []
        for tw in test_twarr:
            idfvec, added = localcounter.idf_vector_of_wordlabels(tw['pos'])
            test_idf_mtx.append(idfvec * np.log(len(added) + 2))
        test_loss = classifier.predict(test_idf_mtx)
        print('predictions:')
        for i, e in enumerate(test_loss[0]):
            print(e, test_twarr[i]['text'], localcounter.idf_vector_of_wordlabels(test_twarr[i]['pos'])[1], '\n')
        print('test_loss', test_loss[1])
        print('mean:', np.mean(test_loss[0]), 'var:', np.var(test_loss[0]))
        
        # test_twarr = self.not_added_twarr[int(len(self.not_added_twarr) * 98 / 100):]
        # test_idf_mtx = []
        # for tw in test_twarr:
        #     idfvec, added = localcounter.idf_vector_of_wordlabels(tw['pos'])
        #     test_idf_mtx.append(idfvec * np.log(len(added)))
        # test_loss = classifier.predict(test_idf_mtx)
        # print('\n\n\npredictions:')
        # for i, e in enumerate(test_loss[0]):
        #     print(e, test_twarr[i]['pos'])
        # print('test_loss', test_loss[1])
    
    def is_file_of_query_date(self, file):
        for seed_query in self.seed_query_list:
            tw_ymd = seed_query.time_of_tweet(file, source='filename')
            if seed_query.is_time_desired(tw_ymd):
                return True
        return False
    
    def parse_ner_text_into_wordlabels(self, ner_text):
        # wordlabels = [('word_0', 'entity extraction word_0', 'pos word_0'), ('word_1', ...), ...]
        ner_words = re.split('\s', ner_text)
        wordlabels = []
        for ner_word in ner_words:
            if ner_word is '':
                continue
            wordlabels.append(self.parse_ner_word_into_labels(ner_word, slash_num=2))
        return wordlabels
    
    def parse_ner_word_into_labels(self, ner_word, slash_num):
        """
        Split a word into array by '/' searched from the end of the word to its begin.
        :param ner_word: With pos labels.
        :param slash_num: Specifies the number of "/" in the pos word.
        :return: Assume that slash_num=2, "qwe/123"->["qwe","123"], "qwe/123/zxc"->["qwe","123","zxc"],
                                  "qwe/123/zxc/456"->["qwe/123","zxc","456"],
        """
        res = []
        over = False
        for i in range(slash_num):
            idx = ner_word.rfind('/') + 1
            res.insert(0, ner_word[idx:])
            ner_word = ner_word[0:idx - 1]
            if idx == 0:
                over = True
                break
        if not over:
            res.insert(0, ner_word)
        return res
    
    def remove_noneword_from_wordlabels(self, wordlabels):
        for idx, wordlabel in enumerate(wordlabels):
            if re.search('^[^a-zA-Z0-9]+$', wordlabel[0]) is not None:
                del wordlabels[idx]
        return wordlabels
    
    def dump_query_list_results(self, seed_path):
        theme = self.theme if self.theme else 'default_theme'
        theme_path = FileIterator.append_slash_if_necessary(seed_path + theme)
        if os.path.exists(theme_path):
            shutil.rmtree(theme_path)
        os.makedirs(theme_path)
        self.localcounter.dump_worddict(theme_path + 'dict.dic')
        self.globalcounter.dump_worddict(theme_path + 'dict.dic')
        for seed_query in self.seed_query_list:
            seed_query.dump_query_results(theme_path)


def parse_files_in_path(json_path, *args, **kwargs):
    # On finding a path containing XXX.json, read every file in it
    seed_parser = kwargs['seed_parser']
    subfiles = FileIterator.listchildren(json_path, children_type='file')
    for subfile in subfiles:
        if not subfile.endswith('.sum'):
            continue
        json_file = json_path + subfile
        seed_parser.read_tweet_from_json_file(json_file)
    seed_parser.sort_idf_of_added_tweet(7, 6)


def parse_querys(data_path, seed_path):
    get_ner_service_pool().start(pool_size=8, classify=False, pos=True)
    since = ['2016', '11', '10']
    until = ['2016', '11', '15']
    seed_parser = SeedParser([
        # [{'all_of': ['obama'], 'none_of': ['trump']}, since, until],
        # [{'all_of': ['trump'], 'none_of': ['obama']}, since, until],
        [{'any_of': ['terror', 'attack', 'isis']}, since, until],
        # [{'all_of': ['trump']}, since, until],
        # [{'all_of': ['obama', 'trump']}, since, until],
        # [{'any_of': ['obama', 'trump'], }, since, until],
    ], theme='terrorist')
    FileIterator.iterate_file_tree(FileIterator.append_slash_if_necessary(data_path),
                                   parse_files_in_path, seed_parser=seed_parser)
    # All tweets have been processed at this moment, with queries holding their desired results
    seed_parser.dump_query_list_results(FileIterator.append_slash_if_necessary(seed_path))
    get_ner_service_pool().end()

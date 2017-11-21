import re

import ArrayUtils
import TweetKeys
import FileIterator
from NerServiceProxy import get_ner_service_pool
from EventClassifier import EventClassifier
from WordFreqCounter import WordFreqCounter

import numpy as np


class EventTrainer:
    def __init__(self):
        self.seed_twarr = self.unlb_twarr = self.cntr_twarr = self.classifier = self.freqcounter = None
    
    @staticmethod
    def start_ner_service(pool_size=8, classify=True, pos=True):
        get_ner_service_pool().start(pool_size, classify, pos)
    
    @staticmethod
    def end_ner_service():
        get_ner_service_pool().end()
    
    def train_and_test(self, seed_twarr, unlb_twarr, cntr_twarr):
        part_arr = (9, 1)
        seed_train, seed_test = ArrayUtils.array_partition(seed_twarr, part_arr)
        cntr_train, cntr_test = ArrayUtils.array_partition(cntr_twarr, part_arr)
        
        # [2.1, 1.4, 1.2, ]:
        param = 1.2
        localcounter, event_classifier, loss = self.train_epoch(seed_train, unlb_twarr, cntr_train, param)
        auc = self.test(localcounter, event_classifier, seed_test, cntr_test)
        print('test - auc:', auc, ' loss', loss, 'vocab', localcounter.vocabulary_size())
        
        self.classifier = event_classifier
        self.freqcounter = localcounter
        return localcounter, event_classifier
    
    def train_epoch(self, seed_train, unlb_train, cntr_train, hyperparam):
        localcounter = WordFreqCounter()
        globalcounter = WordFreqCounter()
        
        localcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in seed_train])
        globalcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in unlb_train])
        globalcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in cntr_train])
        print('\npos pre vocab', localcounter.vocabulary_size(), end=', ')
        localcounter.reserve_word_by_idf_condition(rsv_cond=lambda idf: idf > hyperparam)
        print('pos post vocab', localcounter.vocabulary_size())
        print('neg pre vocab', globalcounter.vocabulary_size(), end=', ')
        globalcounter.reserve_word_by_idf_condition(rsv_cond=lambda idf: idf > hyperparam)
        print('neg post vocab', globalcounter.vocabulary_size())
        localcounter.merge_from(globalcounter)
        print('pos merge vocab', localcounter.vocabulary_size())
        
        seed_train_mtx = localcounter.feature_matrix_of_twarr(seed_train)
        cntr_train_mtx = localcounter.feature_matrix_of_twarr(cntr_train)
        train_mtx = np.concatenate((seed_train_mtx, cntr_train_mtx), axis=0)
        seed_train_lbl = np.array([[1.0]] * len(seed_train), dtype=np.float32)
        cntr_train_lbl = np.array([[0.0]] * len(cntr_train), dtype=np.float32)
        train_lbl = np.concatenate((seed_train_lbl, cntr_train_lbl), axis=0)
        unlb_mtx = localcounter.feature_matrix_of_twarr(unlb_train)
        unlb_lbl = np.array([[0.1]], dtype=np.float32)
        unlbreg_lambda = 0.01
        l2reg_lambda = 0.01
        
        print('seed train:', len(seed_train), 'cntr train:', len(cntr_train), 'unlb:', len(unlb_train))
        print('unlbreg_lambda:', unlbreg_lambda, 'l2reg_lambda:', l2reg_lambda)
        
        classifier = EventClassifier(vocab_size=localcounter.vocabulary_size(), learning_rate=4e-2,
                                     unlbreg_lambda=unlbreg_lambda, l2reg_lambda=l2reg_lambda)
        final_loss = classifier.train_steps(300, train_mtx, train_lbl, unlb_mtx, unlb_lbl, print_loss=True)
        return localcounter, classifier, final_loss
    
    def test(self, freqcounter, classifier, pos_twarr, neg_twarr):
        pos_feature_mtx = freqcounter.feature_matrix_of_twarr(pos_twarr)
        neg_feature_mtx = freqcounter.feature_matrix_of_twarr(neg_twarr)
        pos_preds = classifier.predict(pos_feature_mtx)
        neg_preds = classifier.predict(neg_feature_mtx)
        score_label_pairs = [(s1[0], 1) for s1 in pos_preds] + [(s2[0], 0) for s2 in neg_preds]
        auc = ArrayUtils.roc(score_label_pairs)
        return auc
    
    @staticmethod
    def extract_tw_with_high_freq_entity(twarr, entity_freq=3):
        # extracts tweets which holds high frequency entity from twarr(which is of one day time)
        if TweetKeys.key_wordlabels not in twarr[0]:
            raise ValueError('NER not performed on the twarr.')
        entity_dict = dict()
        for tw in twarr:
            tw['entities'] = dict()
            for wrdlbl in tw[TweetKeys.key_wordlabels]:
                if wrdlbl[1].startswith('O'):
                    continue
                entity_name = wrdlbl[0].strip().lower()
                if entity_name not in entity_dict:
                    entity_dict[entity_name] = {'count': 1}
                else:
                    entity_dict[entity_name]['count'] += 1
                tw['entities'][entity_name] = entity_dict[entity_name]
        for i in range(len(twarr) - 1, -1, -1):
            tw = twarr[i]
            candidate = False
            for entity_key, entity_count in tw['entities'].items():
                if entity_count['count'] >= entity_freq:
                    candidate = True
            if not candidate:
                # print(tw[TweetKeys.key_origintext])
                # print(tw[TweetKeys.key_wordlabels])
                # print('delete', tw['entities'], '\n')
                del twarr[i]
            else:
                # print(tw[TweetKeys.key_origintext])
                # print(tw[TweetKeys.key_wordlabels])
                # print(tw['entities'], '\n')
                tw.pop('entities')
        return twarr
    
    def perform_ner_on_tw_file(self, tw_file, output_to_another_file=False, another_file='./deafult.sum'):
        twarr = FileIterator.load_array(tw_file)
        if not twarr:
            print('No tweets read from file', tw_file)
            return twarr
        twarr = self.twarr_ner(twarr)
        output_file = another_file if output_to_another_file else tw_file
        FileIterator.dump_array(output_file, twarr)
        print('Ner result written into', output_file, ',', len(twarr), 'tweets processed.')
        return twarr
    
    def twarr_ner(self, twarr):
        """
        Perform NER and POS task upon the twarr, inplace.
        :param twarr:
        :return:
        """
        ner_text_arr = get_ner_service_pool().execute_ner_multiple([tw['text'] for tw in twarr])
        if not len(ner_text_arr) == len(twarr):
            raise ValueError("Return line number inconsistent; Error occurs during NER")
        for idx, ner_text in enumerate(ner_text_arr):
            wordlabels = self.parse_ner_text_into_wordlabels(ner_text)
            wordlabels = self.remove_noneword_from_wordlabels(wordlabels)
            twarr[idx][TweetKeys.key_wordlabels] = wordlabels
        return twarr
    
    def parse_ner_text_into_wordlabels(self, ner_text):
        # wordlabels = [('word_0', 'entity extraction word_0', 'pos word_0'), ('word_1', ...), ...]
        words = re.split('\s', ner_text)
        wordlabels = list()
        for word in words:
            if word == '':
                continue
            wordlabels.append(self.parse_ner_word_into_labels(word, slash_num=2))
        return wordlabels
    
    def parse_ner_word_into_labels(self, ner_word, slash_num):
        """
        Split a word into array by '/' searched from the end of the word to its begin.
        :param ner_word: With pos labels.
        :param slash_num: Specifies the number of "/" in the pos word.
        :return: Assume that slash_num=2, "qwe/123"->["qwe","123"], "qwe/123/zxc"->["qwe","123","zxc"],
                                  "qwe/123/zxc/456"->["qwe/123","zxc","456"],
        """
        res = list()
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

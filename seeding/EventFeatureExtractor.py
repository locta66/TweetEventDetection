import re

import ArrayUtils
import TweetKeys
import FileIterator
from NerServiceProxy import get_ner_service_pool
from EventClassifier import EventClassifier
from WordFreqCounter import WordFreqCounter

import numpy as np


class EventFeatureExtractor:
    def __init__(self):
        self.seed_twarr = self.unlb_twarr = self.cntr_twarr = None
        self.seed_train = self.seed_validate = self.seed_test = None
        self.cntr_train = self.cntr_validate = self.cntr_test = None
        self.event_classifier = None
    
    def start_ner_service(self, pool_size=8, classify=True, pos=True):
        get_ner_service_pool().start(pool_size, classify, pos)
    
    def end_ner_service(self):
        get_ner_service_pool().end()
    
    def load_seed_twarr(self, seed_tw_file):
        seed_twarr = FileIterator.load_array(seed_tw_file)
        # for i in range(len(seed_twarr)-1, -1, -1):
        #     tw = seed_twarr[i]
        #     if tw[TweetKeys.key_tagtimes] == 0 or tw[TweetKeys.key_ptagtime] <= tw[TweetKeys.key_ntagtime]:
        #         del seed_twarr[i]
        self.seed_twarr = seed_twarr
        return seed_twarr
    
    def load_unlb_twarr(self, unlb_tw_file):
        self.unlb_twarr = FileIterator.load_array(unlb_tw_file)
        return self.unlb_twarr
    
    def load_cntr_twarr(self, cntr_tw_file):
        self.cntr_twarr = FileIterator.load_array(cntr_tw_file)
        return self.cntr_twarr
    
    def split_seed_twarr(self, part_arr=(7, 3)):
        self.seed_train, self.seed_test = ArrayUtils.array_partition(self.seed_twarr, part_arr)
    
    def split_cntr_twarr(self, part_arr=(7, 3)):
        self.cntr_train, self.cntr_test = ArrayUtils.array_partition(self.cntr_twarr, part_arr)
    
    def train_and_test(self, localcounter, global_counter):
        self.split_seed_twarr()
        self.split_cntr_twarr()
        
        for sort_exponent in range(5, 8):
            auc_sum = 0
            for i in range(5):
                l, e, auc = self.train_and_test_epoch_for_exponent(sort_exponent)
                auc_sum += auc
            print('mean auc:', auc_sum / 5)
            #     epoch_vector += np.array([np.mean(rtl), np.mean(rcl), np.mean(tl), np.mean(cl)], dtype=np.float32)
            # epoch_vector /= 1
            # print('\nexponent', sort_exponent, 're_test_loss ave', epoch_vector[0],
            #       're_cntr_loss ave', epoch_vector[1], 'test_loss ave, ', epoch_vector[2],
            #       'cntr_loss ave, ', epoch_vector[3], '\n--------------------\n')
            
            # for idx, predict in enumerate(test_loss[0]):
            #     if predict < 0.5:
            #         print('se', self.seed_train[idx][TweetKeys.key_cleantext])
            # print('\n--------------------------\n')
            # for idx, predict in enumerate(cntr_loss[0]):
            #     if predict > 0.5:
            #         print('cn', self.cntr_train[idx][TweetKeys.key_cleantext])
        return localcounter, self.event_classifier
    
    def train_and_test_epoch_for_exponent(self, exponent):
        localcounter = WordFreqCounter()
        
        localcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in self.seed_train])
        localcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in self.cntr_train])
        print('\npre vocabulary', localcounter.vocabulary_size(), end=',')
        localcounter.reserve_word_by_idf_threshold(rsv_cond=lambda idf: idf < exponent)
        print(' post vocabulary', localcounter.vocabulary_size())
        
        event_classifier = EventClassifier(vocab_size=localcounter.vocabulary_size(), learning_rate=5e-1,
                                           unlbreg_lambda=0.2, l2reg_lambda=0.1)
        
        seed_train_mtx = self.feature_matrix_of_twarr(self.seed_train, localcounter)
        cntr_train_mtx = self.feature_matrix_of_twarr(self.cntr_train, localcounter)
        seed_train_lbl = np.array([[1.0]] * len(self.seed_train), dtype=np.float32)
        cntr_train_lbl = np.array([[0.0]] * len(self.cntr_train), dtype=np.float32)
        train_mtx = np.concatenate((seed_train_mtx, cntr_train_mtx), axis=0)
        train_lbl = np.concatenate((seed_train_lbl, cntr_train_lbl), axis=0)
        # print('seed instance:', len(self.seed_train), 'cntr instance:', len(self.cntr_train))
        event_classifier.train_steps(1200, train_mtx, train_lbl, None, None, print_loss=False)
        
        seed_test_mtx = self.feature_matrix_of_twarr(self.seed_test, localcounter)
        cntr_test_mtx = self.feature_matrix_of_twarr(self.cntr_test, localcounter)
        # re_test_loss = event_classifier.test(seed_train_mtx, [[1.0]] * len(seed_train_mtx))
        # re_cntr_loss = event_classifier.test(cntr_train_mtx, [[0.0]] * len(cntr_train_mtx))
        test_res = event_classifier.test(seed_test_mtx, [[1.0]] * len(seed_test_mtx))
        cntr_res = event_classifier.test(cntr_test_mtx, [[0.0]] * len(cntr_test_mtx))
        # print('seed test:', len(self.seed_test), 'cntr test:', len(self.cntr_test))
        
        scoreidx = 0
        probidx = 1
        lossidx = 2
        
        score_label_pairs = list()
        for score in test_res[scoreidx]:
            score_label_pairs.append([score[0], 1])
        for score in cntr_res[scoreidx]:
            score_label_pairs.append([score[0], 0])
        auc = ArrayUtils.roc_auc(score_label_pairs)
        print('auc', auc)
        return localcounter, event_classifier, auc
    
    def feature_matrix_of_twarr(self, twarr, wordfreqcounter):
        mtx = list()
        for tw in twarr:
            idfvec, added, num_entity = wordfreqcounter.wordlabel_vector(tw[TweetKeys.key_wordlabels])
            mtx.append(idfvec * np.log(len(added) + 2) * np.log(num_entity + 2))
        return np.array(mtx)
    
    def perform_ner_on_tw_file(self, tw_file, output_to_another_file=False, another_file='./deafult.sum'):
        twarr = FileIterator.load_array(tw_file)
        if not twarr:
            print('No tweets read from file', tw_file)
            return twarr
        # elif TweetKeys.key_wordlabels in twarr[0]:
        #     print('Ner already done for', tw_file)
        #     return twarr
        twarr = self.twarr_ner(twarr)
        output_file = another_file if output_to_another_file else tw_file
        FileIterator.dump_array(output_file, twarr)
        print('Ner result written into', output_file, ',', len(twarr), 'tweets processed.')
        return twarr
    
    def twarr_ner(self, twarr):
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
        ner_words = re.split('\s', ner_text)
        wordlabels = []
        for ner_word in ner_words:
            if ner_word == '':
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



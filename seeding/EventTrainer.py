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
        self.seed_twarr = self.unlb_twarr = self.cntr_twarr = None
        # self.seed_train = self.seed_validate = self.seed_test = None
        # self.unlb_train = self.unlb_validate = self.unlb_test = None
        # self.cntr_train = self.cntr_validate = self.cntr_test = None
    
    def start_ner_service(self, pool_size=8, classify=True, pos=True):
        get_ner_service_pool().start(pool_size, classify, pos)
    
    def end_ner_service(self):
        get_ner_service_pool().end()
    
    def set_seed_twarr(self, seed_twarr):
        self.seed_twarr = seed_twarr
    
    def set_unlb_twarr(self, unlb_twarr):
        self.unlb_twarr = unlb_twarr
    
    def set_cntr_twarr(self, cntr_twarr):
        self.cntr_twarr = cntr_twarr
    
    def train_and_test(self, seed_twarr, unlb_twarr, cntr_twarr):
        part_arr = (7, 3)
        seed_train, seed_test = ArrayUtils.array_partition(seed_twarr, part_arr)
        # ArrayUtils.array_partition(unlb_twarr, (7, 3))
        cntr_train, cntr_test = ArrayUtils.array_partition(cntr_twarr, part_arr)
        
        res_list = list()
        # for sort_exponent in [2.1, 1.4, 1.2, ]:
        for sort_exponent in [1.25, ]:
            auc_sum = 0
            retry = 1
            for i in range(retry):
                (l, c, loss) = self.train_epoch(seed_train, unlb_twarr, cntr_twarr, sort_exponent)
                auc = self.test(l, c, seed_test, cntr_test)
                res_list.append((l, c, loss, auc))
                auc_sum += auc
            print('mean auc:', auc_sum / retry)
        
        res_list = sorted(res_list, key=lambda t: t[-1], reverse=True)
        localcounter = res_list[0][0]
        event_classifier = res_list[0][1]
        print('max:auc', res_list[0][2], ' loss', res_list[0][3], 'vocab', localcounter.vocabulary_size())
        return localcounter, event_classifier
    
    def train_epoch(self, seed_twarr, unlb_twarr, cntr_twarr, hyperparam):
        localcounter = WordFreqCounter()
        globalcounter = WordFreqCounter()
        
        localcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in seed_twarr])
        globalcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in unlb_twarr])
        globalcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in cntr_twarr])
        print('\npos pre vocabulary', localcounter.vocabulary_size(), end=', ')
        localcounter.reserve_word_by_idf_condition(rsv_cond=lambda idf: idf > hyperparam)
        print('pos post vocabulary', localcounter.vocabulary_size())
        print('neg pre vocabulary', globalcounter.vocabulary_size(), end=', ')
        globalcounter.reserve_word_by_idf_condition(rsv_cond=lambda idf: idf > hyperparam)
        print('neg post vocabulary', globalcounter.vocabulary_size())
        localcounter.merge_from(globalcounter)
        print('pos merge vocabulary', localcounter.vocabulary_size())
        
        unlb_mtx = localcounter.feature_matrix_of_twarr(unlb_twarr)
        unlb_lbl = np.array([[0.3]], dtype=np.float32)
        seed_train_mtx = localcounter.feature_matrix_of_twarr(seed_twarr)
        cntr_train_mtx = localcounter.feature_matrix_of_twarr(cntr_twarr)
        train_mtx = np.concatenate((seed_train_mtx, cntr_train_mtx), axis=0)
        seed_train_lbl = np.array([[1.0]] * len(seed_twarr), dtype=np.float32)
        cntr_train_lbl = np.array([[0.0]] * len(cntr_twarr), dtype=np.float32)
        train_lbl = np.concatenate((seed_train_lbl, cntr_train_lbl), axis=0)
        print('seed train:', len(seed_twarr), 'cntr train:', len(cntr_twarr))
        
        classifier = EventClassifier(vocab_size=localcounter.vocabulary_size(), learning_rate=4e-2,
                                     unlbreg_lambda=1, l2reg_lambda=0.1)
        final_loss = classifier.train_steps(500, train_mtx, train_lbl, unlb_mtx, unlb_lbl, print_loss=True)
        return localcounter, classifier, final_loss
    
    def test(self, freqcounter, classifier, pos_twarr, neg_twarr):
        pos_feature_mtx = freqcounter.feature_matrix_of_twarr(pos_twarr)
        neg_feature_mtx = freqcounter.feature_matrix_of_twarr(neg_twarr)
        pos_preds = classifier.predict(pos_feature_mtx)
        neg_preds = classifier.predict(neg_feature_mtx)
        score_label_pairs = [(score[0], 1) for score in pos_preds] + [(score[0], 0) for score in neg_preds]
        auc = ArrayUtils.roc(score_label_pairs)
        return auc
    
    # def train_and_test(self):
    #     self.split_seed_twarr()
    #     self.split_cntr_twarr()
    #
    #     res_list = list()
    #     # for sort_exponent in [2.1, 1.4, 1.2, ]:
    #     for sort_exponent in [1.25, ]:
    #         auc_sum = 0
    #         retry = 1
    #         for i in range(retry):
    #             (l, e, auc, loss) = self.train_and_test_epoch_for_exponent(sort_exponent)
    #             res_list.append((l, e, auc, loss))
    #             auc_sum += auc
    #         print('mean auc:', auc_sum / retry)
    #
    #         # for idx, predict in enumerate(test_loss[0]):
    #         #     if predict < 0.5:
    #         #         print('se', self.seed_train[idx][TweetKeys.key_cleantext])
    #         # print('\n--------------------------\n')
    #         # for idx, predict in enumerate(cntr_loss[0]):
    #         #     if predict > 0.5:
    #         #         print('cn', self.cntr_train[idx][TweetKeys.key_cleantext])
    #
    #     res_list = sorted(res_list, key=lambda t: t[2], reverse=True)
    #     localcounter = res_list[0][0]
    #     event_classifier = res_list[0][1]
    #     print('max auc', res_list[0][2], ' and its loss', res_list[0][3], 'vocab',
    #           localcounter.vocabulary_size())
    #     return localcounter, event_classifier
    
    # def train_and_test_epoch_for_exponent(self, exponent):
        # localcounter = WordFreqCounter()
        # globalcounter = WordFreqCounter()
        
        # localcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in self.seed_train])
        # globalcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in self.unlb_twarr])
        # globalcounter.expand_from_wordlabel_array([tw[TweetKeys.key_wordlabels] for tw in self.cntr_train])
        
        # print('\npos pre vocabulary', localcounter.vocabulary_size(), end=', ')
        # localcounter.reserve_word_by_idf_condition(rsv_cond=lambda idf: idf > exponent)
        # print('pos post vocabulary', localcounter.vocabulary_size())
        # print('neg pre vocabulary', globalcounter.vocabulary_size(), end=', ')
        # globalcounter.reserve_word_by_idf_condition(rsv_cond=lambda idf: idf > exponent)
        # print('neg post vocabulary', globalcounter.vocabulary_size())
        # localcounter.merge_from(globalcounter)
        # print('pos merge vocabulary', localcounter.vocabulary_size())
        
        # unlb_mtx = localcounter.feature_matrix_of_twarr(self.unlb_twarr)
        # unlb_lbl = np.array([[0.3]], dtype=np.float32)
        # seed_train_mtx = localcounter.feature_matrix_of_twarr(self.seed_train)
        # seed_train_lbl = np.array([[1.0]] * len(self.seed_train), dtype=np.float32)
        # cntr_train_mtx = localcounter.feature_matrix_of_twarr(self.cntr_train)
        # cntr_train_lbl = np.array([[0.0]] * len(self.cntr_train), dtype=np.float32)
        # train_mtx = np.concatenate((seed_train_mtx, cntr_train_mtx), axis=0)
        # train_lbl = np.concatenate((seed_train_lbl, cntr_train_lbl), axis=0)
        # print('seed train:', len(self.seed_train), 'cntr train:', len(self.cntr_train))
        
        # seed_test_mtx = localcounter.feature_matrix_of_twarr(self.seed_test)
        # cntr_test_mtx = localcounter.feature_matrix_of_twarr(self.cntr_test)
        # print('seed test:', len(self.seed_test), 'cntr test:', len(self.cntr_test))
        
        # classifier = EventClassifier(vocab_size=localcounter.vocabulary_size(), learning_rate=4e-2,
        #                              unlbreg_lambda=1, l2reg_lambda=0.1)
        # final_loss = classifier.train_steps(500, train_mtx, train_lbl, unlb_mtx, unlb_lbl, print_loss=True)
        # test_res = classifier.test(seed_test_mtx, [[1.0]] * len(seed_test_mtx))
        # cntr_res = classifier.test(cntr_test_mtx, [[0.0]] * len(cntr_test_mtx))
        
        # scoreidx = 0
        # score_label_pairs = [(score[0], 1) for score in test_res[scoreidx]]
        # score_label_pairs.extend([(score[0], 0) for score in cntr_res[scoreidx]])
        # auc = ArrayUtils.roc(score_label_pairs)
        # print('auc', auc, 'loss', final_loss)
        # return localcounter, classifier, auc, final_loss
    
    def extract_tw_with_high_freq_entity(self, twarr, entity_freq=3):
        # extracts tweet which holds high frequency entity within a twarr of one day
        if TweetKeys.key_wordlabels not in twarr[0]:
            raise ValueError('NER not performed on the twarr.')
        entity_dict = dict()
        for tw in twarr:
            tw['entities'] = dict()
            for wordlabel in tw[TweetKeys.key_wordlabels]:
                if wordlabel[1].startswith('O'):
                    continue
                entity_name = wordlabel[0].strip().lower()
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

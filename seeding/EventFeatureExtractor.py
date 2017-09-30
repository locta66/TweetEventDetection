import re

import FileIterator
from NerServiceProxy import get_ner_service_pool


class EventFeatureExtractor:
    key_wordlabels = 'pos'
    
    def __init__(self):
        return
    
    def perform_ner_on_tw_file(self, tw_file, output_to_another_file=False, another_file='./deafult.sum'):
        twarr = FileIterator.load_array(tw_file)
        if not twarr:
            print('No tweets read from file', tw_file)
            return
        if self.key_wordlabels in twarr[0]:
            print('Ner already done for', tw_file)
            return
        get_ner_service_pool().start(pool_size=8, classify=True, pos=True)
        twarr = self.twarr_ner(twarr)
        get_ner_service_pool().end()
        output_file = another_file if output_to_another_file else tw_file
        FileIterator.dump_array(output_file, twarr)
        print('Ner result written into', output_file, ',', len(twarr), 'tweets processed.')
    
    def twarr_ner(self, twarr):
        ner_text_arr = get_ner_service_pool().execute_ner_multiple([tw['text'] for tw in twarr])
        if not len(ner_text_arr) == len(twarr):
            raise ValueError("Return line number inconsistent; Error occurs during NER")
        for idx, ner_text in enumerate(ner_text_arr):
            wordlabels = self.parse_ner_text_into_wordlabels(ner_text)
            wordlabels = self.remove_noneword_from_wordlabels(wordlabels)
            twarr[idx][self.key_wordlabels] = wordlabels
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
    
    # def sort_idf_of_added_tweet(self, localidfthreashold=0.0, globalidfthreashold=0.0):
    #     localcounter = self.localcounter
    #     globalcounter = self.globalcounter
    #     print('added docs', len(self.added_twarr))
    #     for tw in self.twarr_ner(self.added_twarr):
    #         localcounter.expand_dict_and_count_df_from_wordlabels(tw['pos'])
    #     print('not added docs', len(self.not_added_twarr))
    #     for tw in self.twarr_ner(self.not_added_twarr):
    #         globalcounter.expand_dict_and_count_df_from_wordlabels(tw['pos'])
    #
    #     print('pre vocabulary', localcounter.vocabulary_size())
    #     localcounter.reserve_word_by_idf_threshold(rsv_cond=lambda idf: idf < localidfthreashold)
    #     print('mid vocabulary', localcounter.vocabulary_size())
    #     globalcounter.reserve_word_by_idf_threshold(rsv_cond=lambda idf: idf < globalidfthreashold)
    #     local_global_common = set(localcounter.vocabulary()).intersection(set(globalcounter.vocabulary()))
    #     localcounter.remove_words(list(local_global_common), updateid=True)
    #     print('post vocabulary', localcounter.vocabulary_size())
    #     print('common word', local_global_common)
    #
    #     # print('\n')
    #     # for i in range(400):
    #     #     print(self.added_twarr[i]['text'])
    #     #     wordlabels = self.added_twarr[i]['pos']
    #     #     v = localcounter.idf_vector_of_wordlabels(wordlabels)
    #     #     for word in [wordlabel[0].lower() for wordlabel in wordlabels]:
    #     #         if not localcounter.is_word_in_dict(word):
    #     #             continue
    #     #         wordid = localcounter.word_id(word)
    #     #         print(word, '\t\t', localcounter.twdict.worddict[word]['idf'], '\t\t', v[wordid])
    #     #     print('\n')
    #
    #     print('start train')
    #     classifier = EventClassifier(vocab_size=localcounter.vocabulary_size(), learning_rate=2e-2,
    #                                  unlbreg_lambda=0.2, l2reg_lambda=0.1)
    #     print('train len', int(len(self.added_twarr) * 8 / 10))
    #     train_twarr = self.added_twarr[int(len(self.added_twarr) * 8 / 10):]
    #     train_idf_mtx = []
    #     for tw in train_twarr:
    #         idfvec, added = localcounter.idf_vector_of_wordlabels(tw['pos'])
    #         train_idf_mtx.append(idfvec * np.log(len(added) + 2))
    #     import time
    #     s = time.time()
    #     classifier.train_steps(500, 1e-5, None, train_idf_mtx, unlby=[[0.2]])
    #     print('training time ', time.time()-s, 's')
    #
    #     print(classifier.get_value(classifier.thetaEb))
    #     # print(classifier.get_value(classifier.thetaEW))
    #
    #     test_twarr = self.added_twarr[0: int(len(self.added_twarr) * 95 / 100)]
    #     test_idf_mtx = []
    #     for tw in test_twarr:
    #         idfvec, added = localcounter.idf_vector_of_wordlabels(tw['pos'])
    #         test_idf_mtx.append(idfvec * np.log(len(added) + 2))
    #     test_loss = classifier.predict(test_idf_mtx)
    #     print('predictions:')
    #     for i, e in enumerate(test_loss[0]):
    #         print(e, test_twarr[i]['text'], localcounter.idf_vector_of_wordlabels(test_twarr[i]['pos'])[1], '\n')
    #     print('test_loss', test_loss[1])
    #     print('mean:', np.mean(test_loss[0]), 'var:', np.var(test_loss[0]))
    #
    #     # test_twarr = self.not_added_twarr[int(len(self.not_added_twarr) * 98 / 100):]
    #     # test_idf_mtx = []
    #     # for tw in test_twarr:
    #     #     idfvec, added = localcounter.idf_vector_of_wordlabels(tw['pos'])
    #     #     test_idf_mtx.append(idfvec * np.log(len(added)))
    #     # test_loss = classifier.predict(test_idf_mtx)
    #     # print('\n\n\npredictions:')
    #     # for i, e in enumerate(test_loss[0]):
    #     #     print(e, test_twarr[i]['pos'])
    #     # print('test_loss', test_loss[1])

import re
import os

import __init__
import FileIterator
from MyDict import MyDict

import numpy as np
from nltk.corpus import stopwords


class WordFreqCounter:
    def __init__(self, capignore=True, worddict=None):
        self.doc_num = 0
        self.capignore = capignore
        self.stopworddict = self.create_stopword_dict()
        
        self.worddict = worddict if worddict else MyDict()
        pos_dict_file = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'posdict.txt'
        self.posdict = MyDict()
        self.posdict.load_worddict(pos_dict_file)
        self.posdict.reset_ids()
    
    def vocabulary_size(self):
        return self.worddict.vocabulary_size() + self.posdict.vocabulary_size()
    
    @staticmethod
    def create_stopword_dict():
        stopdict = {}
        for stopword in stopwords.words('english'):
            stopdict[stopword] = True
        return stopdict
    
    def is_valid_keyword(self, word):
        # Assume that word has been properly processed.
        isword = re.search('^[^A-Za-z]+$', word) is None
        notchar = re.search('^\w$', word) is None
        notstopword = word not in self.stopworddict
        return isword and notchar and notstopword
    
    def is_valid_wordlabel(self, wordlabel):
        notentity = wordlabel[1].startswith('O')
        return notentity
    
    def reset_freq_couter(self):
        self.doc_num = 0
        for word in self.worddict.dictionary:
            self.worddict.dictionary[word]['df'] = 0
            self.worddict.dictionary[word]['idf'] = 0
    
    def calculate_idf(self):
        if self.doc_num == 0:
            raise ValueError('No valid word has been recorded yet.')
        for word in self.worddict.dictionary:
            df = self.worddict.dictionary[word]['df'] + 1
            self.worddict.dictionary[word]['idf'] = np.log(self.doc_num / df)
    
    def wordlabel_vector(self, wordlabels):
        added_word = dict()
        word_vector = np.array([0] * self.worddict.vocabulary_size(), dtype=np.float32)
        pos_vector = np.array([0] * self.posdict.vocabulary_size(), dtype=np.float32)
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            
            if word in added_word:
                continue
            added_word[word] = True
            
            if not self.worddict.is_word_in_dict(word):
                pos_tag = wordlabel[2]
                pos_vector[self.posdict.word_2_id(pos_tag)] += 1
            else:
                wordid = self.worddict.word_2_id(word)
                word_vector[wordid] = self.worddict.dictionary[word]['idf']
        return np.concatenate([word_vector, pos_vector]), sorted(added_word.keys()), \
            sum([1 for wordlabel in wordlabels if not self.is_valid_wordlabel(wordlabel)])
    
    def expand_dict_and_count_df_from_wordlabel(self, wordlabels):
        added_word = {}
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word:
                    continue
                added_word[word] = True
                # "word" is now neither entity nor invalid keyword or duplicated word now
                self.worddict.expand_dict_from_word(word)
                if 'df' not in self.worddict.dictionary[word]:
                    self.worddict.dictionary[word]['df'] = 1
                else:
                    self.worddict.dictionary[word]['df'] += 1
        self.doc_num += 1
    
    def expand_from_wordlabel_array(self, wordlabel_arr):
        for wordlabel in wordlabel_arr:
            self.expand_dict_and_count_df_from_wordlabel(wordlabel)
        self.worddict.reset_ids()
    
    def reserve_word_by_idf_threshold(self, rsv_cond):
        self.calculate_idf()
        for word in sorted(self.worddict.dictionary.keys()):
            word_idf = self.worddict.dictionary[word]['idf']
            if not rsv_cond(word_idf):
                self.worddict.remove_word(word)
        self.worddict.reset_ids()
    
    # def reserve_word_by_rank(self, rsv_cond):
    #     from operator import itemgetter
    #     self.calculate_idf()
    #     idf_ranked_list = sorted(self.worddict.items(), key=itemgetter(1), reverse=True)
    #     total = len(idf_ranked_list)
    #     for rank, (word, idf) in enumerate(idf_ranked_list):
    #         if not rsv_cond(rank, total):
    #             self.remove_word(word)

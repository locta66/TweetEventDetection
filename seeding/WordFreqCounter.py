import re
import os

from MyDict import MyDict
import TweetKeys
import PatternUtils as pu

import numpy as np
from nltk.corpus import stopwords
stopdict = set(stopwords.words('english'))


class WordFreqCounter:
    def __init__(self, capignore=True, worddict=None):
        self.doc_num = 0
        self.capignore = capignore
        
        self.worddict = worddict if worddict else MyDict()
        pos_dict_file = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'posdict.txt'
        self.posdict = MyDict()
        self.posdict.load_worddict(pos_dict_file)
        self.posdict.reset_ids()
        self.notional = {'NN': 0, 'NNP': 0, 'NNPS': 0, 'NNS': 0, 'RB': 0, 'RBR': 0, 'RBS': 0,
                         'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, }
        self.verb = {'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, }
    
    def vocabulary_size(self):
        # return self.worddict.vocabulary_size() + self.posdict.vocabulary_size()
        return self.worddict.vocabulary_size()
    
    @staticmethod
    def is_valid_keyword(word):
        if not word:
            return False
        # Assume that word has been properly processed.
        startswithchar = re.search('^[^a-zA-Z#]', word) is None
        notsinglechar = re.search('^\w$', word) is None
        notstopword = word not in pu.stop_words
        return startswithchar and notsinglechar and notstopword
    
    def is_valid_wordlabel(self, wordlabel):
        isnotentity = wordlabel[1].startswith('O')
        return isnotentity
    
    def calculate_idf(self):
        if self.doc_num == 0:
            raise ValueError('No valid word has been recorded yet.')
        for word in self.worddict.dictionary:
            df = self.worddict.dictionary[word]['df']
            self.worddict.dictionary[word]['idf'] = 10 / np.log((self.doc_num + 1) / df)
    
    def feature_matrix_of_twarr(self, twarr):
        mtx = list()
        for tw in twarr:
            idfvec, added, num_entity = self.wordlabel_vector(tw[TweetKeys.key_wordlabels])
            mtx.append(idfvec * (np.log(len(added) + 1) + 1) * (np.log(num_entity + 1) + 1))
            # mtx.append(idfvec)
        return np.array(mtx)
    
    def wordlabel_vector(self, wordlabels):
        added_word_dict = dict()
        word_vector = np.array([0] * self.worddict.vocabulary_size(), dtype=np.float32)
        pos_vector = np.array([0] * self.posdict.vocabulary_size(), dtype=np.float32)
        for wordlabel in wordlabels:
            word = wordlabel[0].lower().strip("#") if self.capignore else wordlabel[0]
            # word = get_root_word(word) if wordlabel[2] in self.verb else word
            # if not wordlabel[0].lower().strip("#") == word:
            #     print(wordlabel[2], wordlabel[0].lower().strip("#"), '->', word)
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            if word in added_word_dict:
                continue
            added_word_dict[word] = True
            if not self.worddict.is_word_in_dict(word):
                pos_tag = wordlabel[2]
                pos_vector[self.posdict.word_2_id(pos_tag)] += 1
            else:
                wordid = self.worddict.word_2_id(word)
                word_vector[wordid] = self.worddict.dictionary[word]['idf']
        added_word = sorted(added_word_dict.keys())
        added_entity = sorted([1 for w in wordlabels if not self.is_valid_wordlabel(w)])
        return word_vector, added_word, len(added_entity)
        # return np.concatenate([word_vector, pos_vector]), added_word, len(added_entity)
    
    def expand_dict_and_count_df_from_wordlabel(self, wordlabels):
        added_word_dict = dict()
        for wordlabel in wordlabels:
            word = wordlabel[0].lower().strip("#") if self.capignore else wordlabel[0]
            # word = get_root_word(word) if wordlabel[2] in self.verb else word
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word_dict:
                    continue
                added_word_dict[word] = True
                # "word" is now neither entity nor invalid keyword or duplicated word by now
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
    
    def reserve_word_by_idf_condition(self, rsv_cond):
        self.calculate_idf()
        for word in list(self.worddict.dictionary.keys()):
            word_idf = self.worddict.dictionary[word]['idf']
            if not rsv_cond(word_idf):
                self.worddict.remove_word(word)
        self.worddict.reset_ids()
    
    def merge_from(self, othercounter):
        thisdict = self.worddict.dictionary
        otherdict = othercounter.worddict.dictionary
        for otherword, otherwordattr in otherdict.items():
            if otherword not in thisdict:
                thisdict[otherword] = otherwordattr
                thisdict[otherword]['idf'] /= 5
    
    def most_common_words(self, rank):
        wordnum = self.worddict.vocabulary_size()
        if 0 < rank < 1:
            top_k = wordnum * rank
        elif rank > 1 and type(rank) is int:
            top_k = rank
        else:
            raise ValueError('rank is not a valid number' + str(rank))
        dic = self.worddict.dictionary
        return sorted(dic.keys(), key=lambda w: dic[w]['idf'])[:top_k]
    
    def dump_worddict(self, dict_file, overwrite=True):
        self.worddict.dump_worddict(dict_file, overwrite)
    
    def load_worddict(self, dict_file):
        self.worddict.load_worddict(dict_file)

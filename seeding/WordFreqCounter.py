import re
import __init__
from TweetDict import TweetDict
import FileIterator

from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf


class WordFreqCounter(TweetDict):
    def __init__(self, capignore=True, worddict=None):
        TweetDict.__init__(self)
        self.doc_num = 0
        self.capignore = capignore
        self.stopworddict = self.create_stopword_dict()
        self.worddict = worddict if worddict else self.worddict
    
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
        for word in self.worddict:
            self.worddict[word]['df'] = 0
            self.worddict[word]['idf'] = 0
    
    def calculate_idf(self):
        if self.doc_num == 0:
            raise ValueError('No valid word has been recorded yet.')
        for word in self.worddict:
            df = self.worddict[word]['df'] + 1
            self.worddict[word]['idf'] = np.log(self.doc_num / df)
    
    def expand_dict_and_count_df_from_wordlabels(self, wordlabels):
        added_word = {}
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word:
                    continue
                added_word[word] = True
                # "word" is now neither entity nor invalid keyword or duplicated word
                self.expand_dict_from_word(word)
                if 'df' not in self.worddict[word]:
                    self.worddict[word]['df'] = 1
                else:
                    self.worddict[word]['df'] += 1
        self.doc_num += 1
    
    def remove_word_by_idf_threshold(self, rmv_cond):
        self.calculate_idf()
        for word in sorted(self.worddict.keys()):
            word_idf = self.worddict[word]['idf']
            if not rmv_cond(word_idf):
                self.remove_words([word])
    
    def idf_vector_of_wordlabels(self, wordlabels):
        added_word = {}
        vector = np.zeros(self.vocabulary_size(), dtype=np.float32)
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word:
                    continue
                added_word[word] = True
                wordid = self.word_id(word)
                if wordid:
                    vector[wordid] = self.worddict[word]['idf']
        return vector
    
    def dump_worddict(self, dict_file):
        FileIterator.dump_array(dict_file, [self.worddict])


# def test_train():
#     fc = FreqCounter()
#     for w in ['me', 'you', 'he', 'never', 'give', 'up', 'not', 'only', 'and', 'put', ]:
#         fc.expand_dict_from_word(w)
#     fc.reset_freq_couter()
#     fc.count_df_from_wordarray('he never give me'.split(' '))
#     fc.count_df_from_wordarray('you shall put and on yourself'.split(' '))
#     fc.count_df_from_wordarray('i should not put and on yourself'.split(' '))
#     fc.count_df_from_wordarray('have you ever been defeated'.split(' '))
#     fc.calculate_idf()
#     idfmatrix = list()
#     idfmatrix.append(fc.idf_vector_of_wordarray('he never give me'.split(' ')))
#     idfmatrix.append(fc.idf_vector_of_wordarray('you shall put and on yourself'.split(' ')))
#     idfmatrix.append(fc.idf_vector_of_wordarray('i should not put and on yourself'.split(' ')))
#     idfmatrix.append(fc.idf_vector_of_wordarray('have you ever been defeated'.split(' ')))
#     fc.supervised_loss(idfmatrix, [[0]]*len(idfmatrix))
#
#
# if __name__ == "__main__":
#     test_train()

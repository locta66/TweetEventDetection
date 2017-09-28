import re
import __init__
import FileIterator
from TweetDict import TweetDict

import numpy as np
from nltk.corpus import stopwords


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
    
    def reserve_word_by_idf_threshold(self, rsv_cond):
        self.calculate_idf()
        for word in sorted(self.worddict.keys()):
            word_idf = self.worddict[word]['idf']
            if not rsv_cond(word_idf):
                self.remove_word(word)
    
    def reserve_word_by_rank(self, rsv_cond):
        from operator import itemgetter
        self.calculate_idf()
        idf_ranked_list = sorted(self.worddict.items(), key=itemgetter(1), reverse=True)
        total = len(idf_ranked_list)
        for rank, (word, idf) in enumerate(idf_ranked_list):
            if not rsv_cond(rank, total):
                self.remove_word(word)
    
    def idf_vector_of_wordlabels(self, wordlabels):
        added_word = {}
        vector = np.array([0]*self.vocabulary_size(), dtype=np.float32)
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word:
                    continue
                added_word[word] = True
                wordid = self.word_2_id(word)
                if wordid:
                    vector[wordid] = self.worddict[word]['idf']
        return vector, sorted(added_word.keys())
    
    def dump_worddict(self, dict_file, overwrite=False):
        FileIterator.dump_array(dict_file, [self.worddict], overwrite)
    
    def load_worddict(self, dict_file):
        FileIterator.load_array(dict_file)

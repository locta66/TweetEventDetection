from copy import deepcopy
import pandas as pd
import utils.function_utils as fu


K_ID, K_FREQ, K_DELIMITER = 'ID', 'FREQ', '\t'


class IdFreqDict:
    def __init__(self):
        self._word2id = dict()
        self._id2word = dict()
        self._freq_sum = 0
        self._word_freq_enum = None
    
    def __contains__(self, word): return self.has_word(word)
    
    def clear(self):
        self._word2id.clear()
        self._id2word.clear()
        self._freq_sum = 0
        self._word_freq_enum = None
    
    def has_word(self, word): return word in self._word2id
    
    def vocabulary(self): return list(self._word2id.keys())
    
    def vocabulary_size(self): return self._word2id.__len__()
    
    def word2id(self, word): return self._word2id[word][K_ID]
    
    def id2word(self, _id): return self._id2word[_id]
    
    def freq_of_word(self, word): return self._word2id[word][K_FREQ]
    
    def get_freq_sum(self): return self._freq_sum
    
    def calc_freq_sum(self):
        self._freq_sum = sum([freq for word, freq in self.word_freq_enumerate(newest=True)])
        return self._freq_sum
    
    def reset_id(self):
        for idx, word in enumerate(self.vocabulary()):
            self._word2id[word][K_ID] = idx
            self._id2word[idx] = word
    
    def drop_word(self, word):
        if self.has_word(word):
            self._freq_sum -= self.freq_of_word(word)
            return self._word2id.pop(word)
        return None
    
    def drop_words(self, words):
        for word in words:
            self.drop_word(word)
    
    def drop_words_by_condition(self, condition):
        if type(condition) is int:
            min_freq = condition
            def condition(_, word_freq): return word_freq < min_freq
        if not callable(condition):
            raise ValueError('condition is not int or callable:' + str(condition))
        for word in list(self.vocabulary()):
            if condition(word, self.freq_of_word(word)):
                self.drop_word(word)
        self.reset_id()
    
    def count_word(self, word, freq=None):
        freq = 1 if freq is None else freq
        if self.has_word(word):
            self._word2id[word][K_FREQ] += freq
        else:
            self._word2id[word] = {K_FREQ: freq}
        self._freq_sum += freq
    
    def uncount_word(self, word, freq=None):
        if not self.has_word(word):
            raise ValueError('non existing word "{}"'.format(word))
        freq = 1 if freq is None else freq
        self._word2id[word][K_FREQ] -= freq
        self._freq_sum -= freq
        post_freq = self.freq_of_word(word)
        if post_freq == 0:
            self.drop_word(word)
        elif post_freq < 0:
            raise ValueError('word {} freq {} less than 0'.format(word, post_freq))
    
    def word_freq_enumerate(self, newest=True):
        if self._word_freq_enum is None or newest:
            self._word_freq_enum = [(word, self._word2id[word][K_FREQ]) for word in self.vocabulary()]
        return self._word_freq_enum
    
    def merge_freq_from(self, other_id_freq_dict):
        pre_freq = self._freq_sum
        for other_word, other_freq in other_id_freq_dict.word_freq_enumerate():
            self.count_word(other_word, other_freq)
        return self._freq_sum - pre_freq
    
    def drop_freq_from(self, other_id_freq_dict):
        pre_freq = self._freq_sum
        for other_word, other_freq in other_id_freq_dict.word_freq_enumerate():
            self.uncount_word(other_word, other_freq)
        return pre_freq - self._freq_sum
    
    def dump_dict(self, file_name):
        self.reset_id()
        for word in self.vocabulary():
            if type(word) is not str:
                self.drop_word(word)
        fu.dump_array(file_name, [K_DELIMITER.join(['{}'] * 3).
                      format(word.strip(), int(self.freq_of_word(word)), int(self.word2id(word)))
                                  for word in sorted(self.vocabulary()) if type(word) is str])
    
    def load_dict(self, file_name):
        self.clear()
        for line in fu.load_array(file_name):
            word, freq, _id = line.split(K_DELIMITER)
            self._word2id[word] = {K_FREQ: int(freq), K_ID: int(_id)}
        self.calc_freq_sum()
    
    # def _old_load_dict(self, file_name):
    #     self._word2id = pd.DataFrame().from_csv(file_name).T.to_dict()
    #
    # def _old_dump_dict(self, file_name):
    #     self.word_table().to_csv(file_name, chunksize=self.vocabulary_size())
    #
    # def word_table(self):
    #     self.reset_id()
    #     df = pd.DataFrame(data=self._word2id).T
    #     if self.vocabulary_size() == 0:
    #         return df
    #     for col in [K_ID, K_FREQ]:
    #         df[col] = df[col].astype(int)
    #     return df

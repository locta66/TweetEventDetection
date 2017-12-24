import pandas as pd


class IdFreqDict:
    key_id = 'id'
    key_freq = 'freq'
    
    def __init__(self):
        self._word2id, self._id2word = dict(), dict()
        self.word_freq_enum = None
        self.clear()
    
    def clear(self):
        self._word2id.clear()
        self._id2word.clear()
        self.word_freq_enum = None
    
    def vocabulary(self):
        return sorted(self._word2id.keys())
    
    def vocabulary_size(self):
        return self._word2id.__len__()
    
    def count_word(self, word):
        if word in self._word2id:
            self._word2id[word][self.key_freq] += 1
        else:
            self._word2id[word] = {self.key_freq: 1}
    
    def reset_id(self):
        for idx, word in enumerate(self.vocabulary()):
            self._word2id[word][self.key_id] = idx
            self._id2word[idx] = word
    
    def has_word(self, word):
        return word in self._word2id
    
    def word2id(self, word):
        return self._word2id[word][self.key_id]
    
    def id2word(self, _id):
        return self._id2word[_id]
    
    def word_freq_enumerate(self, newest=False):
        if newest or self.word_freq_enum is None:
            self.word_freq_enum = [(word, self._word2id[word][self.key_freq]) for word in
                                   self._word2id.keys()]
        return self.word_freq_enum
    
    def drop_words_by_condition(self, condition):
        if type(condition) is int:
            min_freq = condition
            def condition(word_freq): return word_freq < min_freq
        if not callable(condition):
            raise ValueError('condition is not callable' + str(condition))
        for word in list(self._word2id.keys()):
            if condition(self._word2id[word][self.key_freq]):
                del self._word2id[word]
        self.reset_id()
    
    def merge_freq_from(self, other_id_freq_dict):
        for other_word, other_freq in other_id_freq_dict.word_freq_enumerate(newest=True):
            if not self.has_word(other_word):
                self._word2id[other_word] = {self.key_freq: other_freq}
            else:
                self._word2id[other_word][self.key_freq] += other_freq
    
    def drop_fre_from(self, other_id_freq_dict):
        for other_word, other_freq in other_id_freq_dict.word_freq_enumerate(newest=True):
            if self.has_word(other_word):
                self._word2id[other_word][self.key_freq] -= other_freq
                if self._word2id[other_word][self.key_freq] < 0:
                    raise ValueError('word freq less than 0')
    
    def word_table(self):
        self.reset_id()
        df = pd.DataFrame(data=self._word2id).T
        for col in [self.key_id, self.key_id]:
            df[col] = df[col].astype(int)
        return df
    
    def dump_dict(self, file_name):
        self.word_table().to_csv(file_name)
    
    def load_dict(self, file_name):
        self._word2id = pd.DataFrame().from_csv(file_name).to_dict()
        self.reset_id()

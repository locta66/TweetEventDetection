import re
import json
from Pattern import get_pattern, PatternHolder

from wordsegment import segment


class TweetDict:
    def __init__(self):
        self.idordered = False
        self.worddict = dict()
        self.clear_dict()
        self.reg_list = [('^[^A-Za-z0-9]+$', ''), ('^\W*', ''), ('\W*$', ''), ("'[sS]?\W*$", ''),
                         ('(.+?)\\1{3,}', '\\1'), ('^(.+?)\\1{3,}$', ''), ]
        self.word_patterns = PatternHolder(self.reg_list, capignore=True)
        self.contraction_list = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'),
                                 (r'i\'m', 'i am'), (r'ain[\'\s]t', 'are not'),
                                 (r'hasn\'t', 'has not'), (r'isn\'t', 'is not'),
                                 (r'(\W|^)you[\'\s]?re(\W|$)', '\\1you are\\2'),
                                 (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                                 (r'(\w+)\'ve', '\g<1> have'),  # (r'(\w+)\'s', '\g<1> is'),
                                 (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), ]
        self.contraction_patterns = PatternHolder(self.contraction_list, capignore=True)
    
    def clear_dict(self):
        if self.worddict is not None:
            self.worddict.clear()
        self.idordered = False
    
    def reset_ids(self):
        for idx, word in enumerate(sorted(self.worddict.keys())):
            self.worddict[word]['id'] = idx
        self.idordered = True
    
    def is_word_in_dict(self, word):
        return word in self.worddict
    
    def word_2_id(self, word):
        return self.worddict[word]['id'] if self.is_word_in_dict(word) else None
    
    def vocabulary(self):
        return sorted(self.worddict.keys())
    
    def vocabulary_size(self):
        return len(self.worddict.keys())
    
    def remove_words(self, words, updateid=False):
        for word in words:
            self.remove_word(word)
        if updateid:
            self.reset_ids()
    
    def remove_word(self, word):
        if self.is_word_in_dict(word):
            self.worddict.pop(word)
        self.idordered = False
    
    def expand_dict_from_word(self, word):
        if not word:
            return
        if not self.is_word_in_dict(word):
            self.worddict[word] = dict()
    
    def text_regularization(self, text, removeht=False, seglen=16):
        modified_text = text.strip()
        modified_text = self.contraction_patterns.apply_pattern(modified_text)
        text_seg = []
        for word in re.split('[|\[\](),.^%$!`~+=_\-\s]', modified_text):
            if word is '':
                continue
            isht = word.startswith('#')
            if isht and not removeht:
                text_seg.append(word.strip())
                continue
            word = self.word_patterns.apply_pattern(word)
            if len(word) >= seglen:
                text_seg.extend(segment(word))
            else:
                text_seg.append(word.strip())
        return text_seg

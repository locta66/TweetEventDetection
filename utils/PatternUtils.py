import re
from wordsegment import segment
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


class SubPatternHolder:
    key_pattern = 'ptn'
    key_substitute = 'sub'
    
    def __init__(self, exp_sub_list=None):
        """processing text with reg may require particular order"""
        self.cache_dict = dict()
        self.exp_list = list()
        if exp_sub_list:
            self.update_pattern_dict(exp_sub_list)
    
    def update_pattern_dict(self, exp_sub_list):
        for exp_sub in exp_sub_list:
            if len(exp_sub) == 3:
                exp, sub, cap_ignore = exp_sub
            elif len(exp_sub) == 2:
                (exp, sub), cap_ignore = exp_sub, True
            else:
                raise ValueError('wrong format ' + str(exp_sub))
            if exp not in self.exp_list:
                self.exp_list.append(exp)
            flags = re.I if cap_ignore else 0
            self.cache_dict[exp] = dict()
            self.cache_dict[exp][self.key_pattern] = re.compile(exp, flags=flags)
            self.cache_dict[exp][self.key_substitute] = sub
    
    def apply_patterns(self, text):
        for exp in self.exp_list:
            pattern = self.cache_dict[exp][self.key_pattern]
            sub = self.cache_dict[exp][self.key_substitute]
            text = pattern.sub(sub, text)
        return text


tw_rule_list = [(r'RT\s@?.*?:(\s|$)', ' '), (r'@\w+', ' '), (r'(#.+?)(\s|$)', ' \\1 '),
                (r'https+:\W*/\.*?(\s|$)|https+:(\s|$)', ' '), ]
special_char_list = [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&ndash;', ' '),
                     ('&mdash;', ' '), ('’', '\''), (r'[~`$%^()\-_=+\[\]{}"/|\\:;]', ' ')]
word_rule_list = [('^[^A-Za-z0-9]+$', ''), ('^\W*', ''), ('\W*$', ''), ("'[sS]?\W*$", ''),
                  ('(.+?)\\1{3,}', '\\1'), ('^(.+?)\\1{3,}$', ''), ]
contraction_list = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),
                    (r'ain[\'\s]t', 'are not'), (r'hasn\'t', 'has not'), (r'isn\'t', 'is not'),
                    (r'(\W|^)you[\'\s]?re(\W|$)', '\\1you are\\2'), (r'(\w+)n\'t', '\g<1> not'),
                    (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)\'d', '\g<1> would'),
                    (r'(\w+)\'ve', '\g<1> have'),  (r'(\w+)\'re', '\g<1> are'), ]

tw_rule_patterns = SubPatternHolder(tw_rule_list)
special_patterns = SubPatternHolder(special_char_list)
word_patterns = SubPatternHolder(word_rule_list)
contraction_patterns = SubPatternHolder(contraction_list)

endline_pattern = SubPatternHolder([(r'[\n\r]+$', ''), ])
brkline_pattern = SubPatternHolder([(r'[\n\r]+', '.'), ])
dupspace_pattern = SubPatternHolder([(r'\s\s+', '.'), ])
nonascii_pattern = SubPatternHolder([(r'[^\x00-\x7f]', '.'), ])


def remove_non_ascii(text): return nonascii_pattern.apply_patterns(text)
def remove_redundant_spaces(text): return dupspace_pattern.apply_patterns(text)
def remove_endline(text): return endline_pattern.apply_patterns(text)
def remove_breakline(text): return brkline_pattern.apply_patterns(remove_endline(text))
def split_digit_arr(string): return [s for s in re.split('[^\d]', string, flags=re.I) if re.findall('^\d+$', s)]
def has_azAZ(string): return len(re.findall(r'^[^a-zA-Z]+$', string.strip())) == 0
def is_stop_word(string): return string.strip().lower() in stop_words
def is_char(string): return len(string) == 1


def text_normalization(text):
    pattern_list = [special_patterns, nonascii_pattern, endline_pattern, brkline_pattern,
                    tw_rule_patterns, _, _, dupspace_pattern, ]
    for pattern in pattern_list:
        text = pattern.apply_patterns(text)
    return text

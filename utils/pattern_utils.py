import re
from wordsegment import segment
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


K_PATTERN = 'ptn'
K_SUB = 'sub'


class PatternHolder:
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
            if exp not in self.cache_dict:
                self.exp_list.append(exp)
            flags = re.I if cap_ignore else 0
            self.cache_dict[exp] = dict()
            self.cache_dict[exp][K_PATTERN] = re.compile(exp, flags=flags)
            self.cache_dict[exp][K_SUB] = sub
    
    def apply_patterns(self, text):
        if len(self.exp_list) == 1:
            exp = self.exp_list[0]
            return self.cache_dict[exp][K_PATTERN].sub(self.cache_dict[exp][K_SUB], text)
        for exp in self.exp_list:
            pattern = self.cache_dict[exp][K_PATTERN]
            sub = self.cache_dict[exp][K_SUB]
            text = pattern.sub(sub, text)
        return text


tw_rule_list = [(r'RT\s@?.*?:(\s|$)', ' '), (r'@\w+', ' '), (r'(#.+?)(\s|$)', ' \\1 '),
                (r'https?:\W*/.*?(\s|$)|https?:(\s|$)', ' '), ]
special_char_list = [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&ndash;', ' '),
                     ('&mdash;', ' '), ('[’‘]', '\''), ]
puctuation_list = [(r'[~`$^()\-_=\+\[\]{}"|:;]', ' '), ]
word_rule_list = [('^[^A-Za-z0-9]+$', ''), ('^\W*', ''), ('\W*$', ''), ("'[sS]?\W*$", ''),
                  ('(.+?)\\1{3,}', '\\1'), ('^(.+?)\\1{3,}$', ''), ]
contraction_list = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),
                    (r'ain[\'\s]t', 'are not'), (r'hasn\'t', 'has not'), (r'isn\'t', 'is not'),
                    (r'(\W|^)you[\'\s]?re(\W|$)', '\\1you are\\2'), (r'(\w+)n\'t', '\g<1> not'),
                    (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)\'d', '\g<1> would'),
                    (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'re', '\g<1> are'), ]

tw_rule_patterns = PatternHolder(tw_rule_list)
special_patterns = PatternHolder(special_char_list)
word_patterns = PatternHolder(word_rule_list)
contraction_patterns = PatternHolder(contraction_list)
punc_pattern = PatternHolder(puctuation_list)

endline_pattern = PatternHolder([(r'[\n\r]+$', ''), ])
brkline_pattern = PatternHolder([(r'[\n\r]+', '.'), ])
dupspace_pattern = PatternHolder([(r'\s\s+', ' '), ])
nonascii_pattern = PatternHolder([(r'[^\x00-\x7f]', '.'), ])


def find_pattern(pattern, string): return re.findall(pattern, string)


def search_pattern(pattern, string): return re.search(pattern, string)


def sub_pattern(pattern, repl, string): return re.sub(pattern, repl, string)


def remove_non_ascii(text): return nonascii_pattern.apply_patterns(text)


def remove_redundant_spaces(text): return dupspace_pattern.apply_patterns(text)


def remove_endline(text): return endline_pattern.apply_patterns(text)


def remove_breakline(text): return brkline_pattern.apply_patterns(remove_endline(text))


def split_digit_arr(string): return re.findall('\d+', string)


def has_azAZ(string): return re.search(r'^[a-zA-Z]+$', string.strip()) is not None


def is_empty_string(string): return len(string) == 0 or len(re.findall(r'^\s+$', string)) > 0


def is_stop_word(string): return string.strip().lower() in stop_words


def is_char(string): return len(string) == 1


def tokenize(pattern, string, flags=re.I): return re.findall(pattern, string, flags=flags)


def word_segment(string): return segment(string)


def text_normalization(text):
    pattern_list = [special_patterns, nonascii_pattern, endline_pattern, brkline_pattern,
                    tw_rule_patterns, contraction_patterns, dupspace_pattern, ]
    for pattern in pattern_list:
        text = pattern.apply_patterns(text)
    return text


def is_valid_keyword(word):
    if not word:
        return False
    startswithchar = re.search('^[^a-zA-Z#]', word) is None
    notsinglechar = not is_char(word)
    notstopword = word not in stop_words
    return startswithchar and notsinglechar and notstopword


if __name__ == '__main__':
    # s = 'Bahrain bans all protests in new crackdown. http://t.co/l0AQNtmB'
    # ss = re.sub(r'https?:\W*/.*?(\s|$)|https?:(\s|$)', ' ', s)
    # print(ss)
    print(text_normalization('Bahrain bans all protests in new crackdown. http://t.co/l0AQNtmB'))

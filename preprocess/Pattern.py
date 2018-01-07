import os
import re


class PatternHolder:
    class PatternItem:
        def __init__(self, expression, pattern, substitute):
            self.expression = expression
            self.pattern = pattern
            self.substitute = substitute
    
    def __init__(self, exp_sub_list=None, capignore=False):
        self.capignore = capignore
        self.pattern_list = list()
        if exp_sub_list:
            self.update_list(exp_sub_list, renew=True)
    
    def update_list(self, exp_sub_list, renew=False):
        if renew:
            self.pattern_list.clear()
        for exp, sub in exp_sub_list:
            flags = re.I if self.capignore else 0
            self.pattern_list.append(self.PatternItem(exp, re.compile(exp, flags=flags), sub))
    
    def apply_pattern(self, text):
        for pattern in self.pattern_list:
            text = pattern.pattern.sub(pattern.substitute, text)
        return text


# class Singleton(object):
#     _instance = None
#
#     def __new__(cls, *args, **kw):
#         if not cls._instance:
#             cls._instance = super(Singleton, cls).__new__(cls)
#         return cls._instance


class Pattern:
    def __init__(self):
        self.special_char_pattern = self.rule_pattern = self.abbr_pattern = self.emo_pattern = None
        # self.file_dir = os.path.split(os.path.realpath(__file__))[0] + os.path.sep
        self.file_dir = os.path.abspath(os.path.dirname(__file__)) + os.path.sep
        self.abbr_dict_file = self.file_dir + 'abbrs.txt'
        self.emo_dict_file = self.file_dir + 'emoticons.txt'
        self.endline_pattern = re.compile(r'[\n\r]+$')
        self.breakline_pattern = re.compile(r'[\n\r]+')
        self.redundant_space_pattern = re.compile('\s\s+')
        self.nonascii_pattern = re.compile(r'[^\x00-\x7f]')
        # self.htmltrans_pattern = re.compile('&\w{2,8};')
        # self.ymdh_pattern = re.compile(r'^(\d{4})(\d{2})?(\d{2})?(\d{2})?$')
        self.update()
    
    def update(self):
        self.update_special_characters()
        self.update_rules()
        self.update_abbrs()
        self.update_emoticons()
    
    def update_special_characters(self):
        char_list = [('’', '\'')]
        self.special_char_pattern = PatternHolder(char_list)
    
    def update_rules(self):
        rule_list = [(r'RT\s@?.*?:(\s|$)', ' '), (r'@\w+', ' '), (r'(#.+?)(\s|$)', ' \\1 '),
                     (r'http:\W*//.*?(\s|$)', ' '), (r'https:\W*//.*?(\s|$)', ' '),
                     ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&ndash;', ' '), ('&mdash;', ' '), ]
        self.rule_pattern = PatternHolder(rule_list)
    
    def update_abbrs(self):
        abbr_list = []
        with open(self.abbr_dict_file) as fp:
            for abbr_line in fp:
                abbr_line = self.remove_endline(abbr_line)
                abbr, full = abbr_line.split('\t')
                if not abbr:
                    print('abbrs.txt format error')
                    continue
                abbr_list.append((abbr, full))
        self.abbr_pattern = PatternHolder(abbr_list)
    
    def update_emoticons(self):
        emo_list = []
        with open(self.emo_dict_file) as fp:
            for icon_line in fp:
                if not icon_line:
                    print('emoticons.txt format error')
                    continue
                normalized_emo_icon = self.normalexp_2_regexp(self.remove_endline(icon_line))
                emo_list.append((normalized_emo_icon, ' '))
        emo_list.append((r'[~`$%^()\-_=+\[\]{}"/|\\:;]', ' '))
        self.emo_pattern = PatternHolder(emo_list)
    
    def normalexp_2_regexp(self, normalexp):
        regexp = normalexp
        fbsarr = ["\\", "$", "(", ")", "*", "+", ".", "[", "]", "?", "^", "{", "}", "|"]
        for key in fbsarr:
            if regexp.find(key) >= 0:
                regexp = regexp.replace(key, '\\' + key)
        return regexp
    
    def normalization(self, text):
        text = self.special_char_pattern.apply_pattern(text)
        text = self.remove_non_ascii(text)
        text = self.remove_breakline(text)
        text = self.rule_pattern.apply_pattern(text)
        text = self.abbr_pattern.apply_pattern(text)
        text = self.emo_pattern.apply_pattern(text)
        text = self.remove_redundant_spaces(text)
        return text
    
    def remove_non_ascii(self, text):
        return self.nonascii_pattern.sub(' ', text)
    
    def remove_redundant_spaces(self, text):
        return self.redundant_space_pattern.sub(' ', text).strip(' ')
    
    def remove_endline(self, text):
        # Remove line breaker like '\r\n'(especially for files from windows) at the end of a string
        return self.endline_pattern.sub('', text)
    
    def remove_breakline(self, text):
        # Remove all line breaker like '\r\n' at any position of a string
        text = self.remove_endline(text)
        return self.breakline_pattern.sub('.', text)
    
    def full_split_nondigit(self, text):
        split = []
        for s in re.split('[^\d]', text):
            if s is not '':
                split.append(s)
        return split


pattern = Pattern()


def get_pattern():
    return pattern


# s = 'RT @bugwannostra: @Louuu_ thx		#FFFFs People power -_-      works	❤signing…		https://t.co/pl2bquE5Az'
# s = 'RT @bugwannost https://baidu.com ra: @Louuu_ thx		#FFFFs People power -_-  https:/    works	❤signing… http:   https:'
# # re.findall(r'[a-zA-Z_\-\']{3,}', s)
# ptn = re.compile(r'https?:\W*/.*?(\s|$)|https?:(\s|$)')
# ptn.sub('_', s)

# Pattern().normalization(s)
# s1 = '@Louuu_ People power see u works❤signing… :-D (https://t.o/pl2u5Az @Louuu_) e.u as me &amp; &amp; \U0001f310'
# Pattern().normalization(s1)

# s1 = '@Louuu_ People power works❤signing… https://t.co/pl2bquE5Az @Louuu_ as me &amp; &amp; '
# reg = re.compile(r'(ig|ing)')
# t = reg.sub(' \\1 ', s1)
# s1
# t

# s="Hit de r()emise  -+_=|/? en forme en France . 「Zumba Fitness」's 『Santa| Que Cumbia』·༺☾ ★ ☽༻"
# ss1 = '@Louuu_ People power see u works❤signing… :-D (https://t.o/pl2u5Az @Louuu_) e.u☹ as me &amp; &amp; \U0001f310'
# ss = "' Kylie Jenner And Tyga' s Sex Tape Got Leaked On Tyga  s Website\u0080"
# Pattern().normalization(ss)
# # s=s.decode('utf8')

# import re
# s = '. .qwe))) asdjk*. '
# p=re.compile(r'[~`$%^()\-_=+\[\]{}"/|\\:;]')
# p.sub(' ', s)


from MyDict import MyDict
import TweetKeys
import FunctionUtils
import time
import numpy as np


class CacheBack:
    key_count = 'count'
    key_tw = 'tw'
    key_timestamp = 'tstamp'
    
    def __init__(self, freqcounter):
        """
        One cluster per cache instance.
        :param freqcounter:
        """
        self.twdict = self.keywords = self.entities_geo = self.entities_non_geo = None
        self.clear_cache()
        self.freqcounter = freqcounter
        self.notional = {'HT': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0, 'NNS': 0, 'RB': 0, 'RBR': 0,
                         'RBS': 0,
                         'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, }
    
    def clear_cache(self):
        self.twdict = dict()
        self.keywords = MyDict()
        self.entities_geo = MyDict()
        self.entities_non_geo = MyDict()
    
    def tweet_number(self):
        return len(self.twdict.keys())
    
    def n_keywords(self):
        return sum([dic[self.key_count] for dic in self.keywords.dictionary.values()])
        # return len(self.keywords.dictionary.keys())
    
    def n_entities_non_geo(self):
        return sum([dic[self.key_count] for dic in self.entities_non_geo.dictionary.values()])
        # return len(self.entities_non_geo.dictionary.keys())
    
    def n_entities_geo(self):
        return sum([dic[self.key_count] for dic in self.entities_geo.dictionary.values()])
        # return len(self.entities_geo.dictionary.keys())
    
    def word_count_incr(self, word, mydict, increment=1):
        if word in mydict.dictionary:
            mydict.dictionary[word][self.key_count] += increment
            if mydict.dictionary[word][self.key_count] <= 0:
                mydict.dictionary.pop(word)
        elif increment > 0:
            mydict.expand_dict_from_word(word)
            mydict.dictionary[word][self.key_count] = increment
    
    def update_from_tw(self, tw, add=True):
        tw_id = tw[TweetKeys.key_id]
        wordlabels = tw[TweetKeys.key_wordlabels]
        increment = 1 if add else -1
        if add:
            if tw_id not in self.twdict:
                self.twdict[tw_id] = {self.key_tw: tw, self.key_timestamp: time.time(), }
            else:
                print("added tweet detected")
        for wordlabel in wordlabels:
            word, nertag, postag = wordlabel
            word = word.lower()
            if not self.freqcounter.is_valid_keyword(word):
                continue
            elif 'geo' in nertag:  # if the wordlabel is a geo entity
                self.word_count_incr(word, self.entities_geo, increment)
            elif not nertag == 'O':  # if the wordlabel is a normal entity
                self.word_count_incr(word, self.entities_non_geo, increment)
            elif postag in self.notional:  # if the wordlabel is a notional word(noun, verb, adverb, etc.)
                self.word_count_incr(word, self.keywords, increment)
        if not add:
            if tw_id in self.twdict:
                self.twdict.pop(tw_id)
    
    def score_with_tw(self, tw, doc_num, event_num, g_vocab, ng_vocab, k_vocab, alpha, beta):
        g_beta0 = g_vocab * beta
        ng_beta0 = ng_vocab * beta
        k_beta0 = k_vocab * beta
        prob = (self.tweet_number() + alpha) / (doc_num - 1 + event_num * alpha)
        num_en_geo = self.n_entities_geo()
        num_en_non_geo = self.n_entities_non_geo()
        num_keyword = self.n_keywords()
        wordlabels = tw[TweetKeys.key_wordlabels][:]
        for i in range(len(wordlabels) - 1, -1, -1):
            wordlabels[i][0] = wordlabels[i][0].lower()
            word = wordlabels[i][0]
            if not self.freqcounter.is_valid_keyword(word):
                del wordlabels[i]
        for i in range(len(wordlabels)):
            if self.entities_geo.is_word_in_dict(word):
                prob *= (self.entities_geo.dictionary[word][self.key_count] + beta) / \
                          (num_en_geo + g_beta0 + i)
            elif self.entities_non_geo.is_word_in_dict(word):
                prob *= (self.entities_non_geo.dictionary[word][self.key_count] + beta) / \
                          (num_en_non_geo + ng_beta0 + i)
            elif self.keywords.is_word_in_dict(word):
                prob *= (self.keywords.dictionary[word][self.key_count] + beta) / \
                          (num_keyword + k_beta0 + i)
        return prob
    
    # def score_with_tw(self, tw, tw_corpus, event_corpus, geo_corpus, non_geo_corpus, keyword_corpus,
    #                   alpha, h_geo, h_non_geo, h_keyword):
    #     wordlabels = tw[TweetKeys.key_wordlabels][:]
    #     for i in range(len(wordlabels) - 1, -1, -1):
    #         wordlabels[i][0] = wordlabels[i][0].lower()
    #         word = wordlabels[i][0]
    #         if not self.freqcounter.is_valid_keyword(word):
    #             del wordlabels[i]
    #     n_m = len(wordlabels)
    #     if n_m == 0:
    #         return 0
    #     numerator = 1
    #     denominator = np.prod([(self.n_entities_geo() - i + geo_corpus * h_geo) *
    #                            (self.n_entities_non_geo() - i + non_geo_corpus * h_non_geo) *
    #                            (self.n_keywords() - i + keyword_corpus * h_keyword)
    #                            for i in range(1, n_m + 1)])
    #     for wordlabel in wordlabels:
    #         word, nertag, postag = wordlabel
    #         if not self.freqcounter.is_valid_keyword(word):
    #             continue
    #         elif self.entities_geo.is_word_in_dict(word):
    #             numerator *= (self.entities_geo.dictionary[word][self.key_count] + h_geo - 1)
    #         elif self.entities_non_geo.is_word_in_dict(word):
    #             numerator *= (self.entities_non_geo.dictionary[word][self.key_count] + h_non_geo - 1)
    #         elif self.keywords.is_word_in_dict(word):
    #             numerator *= (self.keywords.dictionary[word][self.key_count] + h_keyword - 1)
    #     factor = (self.tweet_number() + alpha - 1) / (tw_corpus + event_corpus * alpha)
    #     print('geo_corpus', geo_corpus, ',non_geo_corpus', non_geo_corpus,
    #           ',keyword_corpus', keyword_corpus, ',tw_corpus', tw_corpus, ',event_corpus', event_corpus)
    #     print([(self.n_entities_geo() + i + geo_corpus * h_geo) *
    #            (self.n_entities_non_geo() + i + non_geo_corpus * h_non_geo) *
    #            (self.n_keywords() + i + keyword_corpus * h_keyword)
    #            for i in range(n_m)])
    #     print('n_m', n_m)
    #     print('factor', factor, 'numerator', numerator, 'denominator', denominator)
    #     print('score', factor * numerator / denominator)
    #     print('\n------------------------------\n')
    #     return factor * numerator / denominator
    
    def is_plagiarize(self, outertw, threshold=0.8):
        """
        Judge if there exists plagiarism between outer and inner.
        :param outertw:
        :param threshold:
        :return:
        """
        outerstr = outertw[TweetKeys.key_cleantext].lower()
        if not outerstr:
            return True
        for innertw_and_stamp in self.twdict.values():
            innertw = innertw_and_stamp[self.key_tw]
            innerstr = innertw[TweetKeys.key_cleantext].lower()
            if FunctionUtils.plagiarize_score(outerstr, innerstr) > threshold:
                return True
        return False


class CacheFront(CacheBack):
    def __init__(self, freqcounter):
        CacheBack.__init__(self, freqcounter)
    
    def compare_with_cluster(self, tw):
        pass

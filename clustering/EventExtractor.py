from EventClassifier import EventClassifier
from WordFreqCounter import WordFreqCounter
from Cache import CacheBack, CacheFront
import numpy as np
import pandas as pd


class EventExtractor:
    def __init__(self, dict_file, model_file):
        self.freqcounter = self.classifier = self.filteredtw = None
        self.construct(dict_file, model_file)
        self.cache_back = list()
        self.cache_front = list()
    
    def construct(self, dict_file, model_file):
        self.load_worddict(dict_file)
        self.load_classifier(model_file)
    
    def load_worddict(self, dict_file):
        self.freqcounter = WordFreqCounter()
        self.freqcounter.load_worddict(dict_file)
    
    def load_classifier(self, model_file):
        self.classifier = EventClassifier(vocab_size=self.freqcounter.vocabulary_size(),
                                          learning_rate=0)
        self.classifier.load_params(model_file)
    
    def make_classification(self, twarr):
        return self.classifier.predict(self.freqcounter.feature_matrix_of_twarr(twarr))[0]
    
    def filter_twarr(self, twarr, threshold=0.5):
        return [twarr[idx] for idx, pred in enumerate(self.make_classification(twarr)) if
                pred >= threshold]
    
    def merge_tw_into_cache_back(self, tw):
        if not self.cache_back:
            self.create_cache_with_tw(tw)
            return
        
        geo_corpus = len(
            self.merge_dicts([cache.entities_geo.dictionary for cache in self.cache_back]).keys())
        non_geo_corpus = len(self.merge_dicts(
            [cache.entities_non_geo.dictionary for cache in self.cache_back]).keys())
        keyword_corpus = len(
            self.merge_dicts([cache.keywords.dictionary for cache in self.cache_back]).keys())
        tw_corpus = sum([cache.tweet_number() for cache in self.cache_back])
        event_corpus = len(self.cache_back)
        
        alpha = 1
        h_geo = 0.05
        h_non_geo = 0.2
        h_keyword = 0.5
        
        score_list = [cache.score_with_tw(tw, tw_corpus, event_corpus, geo_corpus, non_geo_corpus,
                                          keyword_corpus, alpha, h_geo, h_non_geo, h_keyword)
                      for cache in self.cache_back]
        
        self.cache_back[0].update_from_tw(tw)
    
    @staticmethod
    def merge_dicts(list_of_dicts):
        res = dict()
        for dictionary in list_of_dicts:
            res.update(dictionary)
        return res
    
    def create_cache_with_tw(self, tw):
        cache = CacheBack(self.freqcounter)
        cache.update_from_tw(tw)
        self.cache_back.append(cache)
    
    def show_weight_of_words(self):
        thetaE = self.classifier.get_theta()[0]
        table = pd.DataFrame.from_dict(self.freqcounter.worddict.dictionary).T
        table.drop(axis=1, labels=['df'], inplace=True)
        table.sort_values(by='idf', ascending=False, inplace=True)
        for index, series in table.iterrows():
            table.loc[index, 'theta'] = thetaE[0][int(series['id'])]
        print(table)

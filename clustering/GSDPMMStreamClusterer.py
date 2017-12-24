import pandas as pd
from IdFreqDict import IdFreqDict


class GSDPMMStreamClusterer:
    def __init__(self):
        self.id2clu = pd.DataFrame()
    
    def set_hyperparams(self, alpha, etap, etac, etav, etah):
        self.paramkeys = ('alpha', 'etap', )
        self.hyperparams = (self.alpha, self.etap, self.etac, self.etav, self.etah) = \
            (alpha, etap, etac, etav, etah)


class ClusterHolder:
    def __init__(self):
        self.m = 0


class ParamHolder:
    def __init__(self, param_value, corpus_dict):
        self.param_value = param_value
        self.corpus_dict = corpus_dict
    
    def get_param(self):
        return self.param_value
    
    def get_param0(self):
        return self.param_value * self.corpus_dict.vocabulary_size()

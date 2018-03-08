import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import spacy

import classifying.fast_text_utils as ftu
import utils.spacy_utils as su
import utils.array_utils as au
import utils.timer_utils as tmu


class ClassifierAddFeature:
    def __init__(self):
        self.nlp = None
        self.ft_model = None
        self.lr_model = None
    
    def get_nlp(self):
        if self.nlp is None:
            self.nlp = spacy.load('en', vector=False)
        return self.nlp
    
    def save_model(self, ft_model_file, lr_model_file):
        ftu.save_model(ft_model_file, self.ft_model)
        joblib.dump(self.lr_model, lr_model_file)
    
    def load_model(self, ft_model_file, lr_model_file):
        self.ft_model = ftu.load_model(ft_model_file)
        self.lr_model = joblib.load(lr_model_file)
    
    def train_ft_model(self, train_file, **kwargs):
        if self.ft_model is None:
            self.ft_model = ftu.FastText()
        self.ft_model.train_supervised(train_file, **kwargs)
    
    def train_lr_model(self, x, y, **kwargs):
        if self.lr_model is None:
            self.lr_model = LogisticRegressionCV(**kwargs)
        self.lr_model.fit(x, y)
    
    def get_fasttext_vector(self, text):
        return self.ft_model.get_sentence_vector(text)
    
    def has_locate_feature(self, doc):
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                return 1
        return -1
    
    def has_keyword_feature(self, text):
        # TODO use re.search
        sensitive_words = {'shooting', 'wounded', 'shots', 'attack', 'shooter', 'wounds', 'dead', 'terrorist',
                           'hurt', 'terror', 'police', 'killed', 'gunman', 'weapon', 'injured', 'attacked',
                           'bomb', 'bombed', 'attacker'}
        for word in text.lower().split():
            word = word.strip()
            if word in sensitive_words:
                return 1
        return -1
    
    def textarr2featurearr(self, textarr):
        docarr = su.textarr_nlp(textarr, self.get_nlp())
        assert len(textarr) == len(docarr)
        vec_arr = list()
        for idx in range(len(textarr)):
            text, doc = textarr[idx], docarr[idx]
            ft_vec = self.get_fasttext_vector(text)
            loc_feature = self.has_locate_feature(doc)
            key_feature = self.has_keyword_feature(text)
            ft_vec = np.append(ft_vec, loc_feature)
            ft_vec = np.append(ft_vec, key_feature)
            vec_arr.append(ft_vec)
        return np.array(vec_arr)
    
    def predict_proba(self, textarr):
        featurearr = self.textarr2featurearr(textarr)
        probaarr = self.lr_model.predict_proba(featurearr)[:, 1]
        return probaarr
    
    def predict_filter(self, textarr, threshold):
        probaarr = self.predict_proba(textarr)
        for idx in range(len(textarr) - 1, -1, -1):
            if probaarr[idx] < threshold:
                textarr.pop(idx)
        return textarr
    
    def train(self, train_file, ft_args, lr_args):
        self.train_ft_model(train_file, **ft_args)
        textarr, labelarr = file2label_text_array(train_file)
        featurearr = self.textarr2featurearr(textarr)
        self.train_lr_model(featurearr, labelarr, **lr_args)
        print('fit over', set(labelarr))
    
    def test(self, test_file):
        textarr, labelarr = file2label_text_array(test_file)
        scorearr = self.predict_proba(textarr)
        auc = au.score(labelarr, scorearr, 'auc')
        print(auc)


label2value = ftu.binary_label2value


def file2label_text_array(file):
    with open(file) as fp:
        lines = fp.readlines()
    print('file line number: {}'.format(len(lines)))
    textarr, labelarr = list(), list()
    for line in lines:
        try:
            label, text = line.strip().split(' ', maxsplit=1)
        except:
            continue
        textarr.append(text)
        labelarr.append(label2value[label])
    print('train text length: {}'.format(len(textarr)))
    return textarr, labelarr


ft_model_file = '/home/nfs/cdong/tw/seeding/Terrorist/model/ft_add_feature_model'
lr_model_file = '/home/nfs/cdong/tw/seeding/Terrorist/model/lr_add_feature_model'


if __name__ == '__main__':
    """ test on cdong data """
    test_file = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/test'
    c = ClassifierAddFeature()
    c.load_model(ft_model_file, lr_model_file)
    tmu.check_time()
    c.test(test_file)
    tmu.check_time(print_func=lambda dt: print('test time: {}s'.format(dt)))
    exit()
    
    """ train and test on yangl data """
    # train_file = '/home/nfs/yangl/event_detection/testdata/fast_gbdt/train.txt'
    # test_file = '/home/nfs/yangl/event_detection/testdata/fast_gbdt/test.txt'
    train_file = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/train'
    test_file = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/test'
    
    ft_args = dict(epoch=100, lr=2, wordNgrams=2, verbose=2, minCount=10)
    lr_args = dict()
    c = ClassifierAddFeature()
    tmu.check_time()
    c.train(train_file, ft_args, lr_args)
    tmu.check_time(print_func=lambda dt: print('train time: {}s'.format(dt)))
    
    c.save_model(ft_model_file, lr_model_file)
    
    tmu.check_time()
    c.test(test_file)
    tmu.check_time(print_func=lambda dt: print('test time: {}s'.format(dt)))

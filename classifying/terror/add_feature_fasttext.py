from classifying.fast_text_utils import FastText
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import spacy


class Classifier(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model = FastText()
        self.fasttext_train_supervised()
        self.nlp = spacy.load('en')
        self.dim = 102
        self.train_feature_matrix = np.zeros((102,))
        self.test_feature_matrix = np.zeros((102,))
        self.train_label = list()
        self.test_label = list()
    
    def get_sentence_vector(self, text):
        return self.model.get_sentence_vector(text)
    
    def fasttext_train_supervised(self):
        self.model.train_supervised(self.train_path)
    
    def predict_prob(self):
        clf = LogisticRegression()
        self.test_feature_matrix = np.delete(self.test_feature_matrix, 0, 0)
        self.train_feature_matrix = np.delete(self.train_feature_matrix, 0, 0)
        # print (self.test_feature_matrix.shape, len(self.test_label))
        # print (self.train_feature_matrix.shape,len(self.train_label))
        print('training begin: ')
        clf.fit(self.train_feature_matrix, self.train_label)
        predict_label = clf.predict_proba(self.test_feature_matrix)[:, 1]
        auc = metrics.roc_auc_score(self.test_label, predict_label)
        print('auc: ', auc)
        precision, recall, thresholds = metrics.precision_recall_curve(self.test_label, predict_label)
        last_idx = 0
        for ref in [i / 10 for i in range(3, 8)]:
            for idx in range(last_idx, len(thresholds)):
                if thresholds[idx] >= ref:
                    print('threshold', round(thresholds[idx], 2), '\tprecision', round(precision[idx], 5)
                    , '\trecall', round(recall[idx], 5))
                    last_idx = idx
                    break

    def feature_vector(self,file,label,sentence_vector,locate_feature,url_feature,sensitive_feature):
        feature = np.append(sentence_vector,locate_feature)
        feature = np.append(feature,sensitive_feature)
        return (feature,len(feature))

    def feature_matrix(self,feature_vector,label,file):
        feature_vector.reshape(102,1)
        assert self.dim == 102
        if file == 'train':
            self.train_feature_matrix = np.row_stack((self.train_feature_matrix,feature_vector))
        else:
            self.test_feature_matrix = np.row_stack((self.test_feature_matrix,feature_vector))

    def locate_feature(self,tweet):
        document = self.nlp(tweet)
        entry_list = list()
        for ent in document.ents:
            entry_list.append(ent.label_)
        label = -1
        if 'GPE' in entry_list:
            label = 1
        return label

    def sensitive_words(self,tweet):
        sensitive_words = ['shooting','wounded','shots','attack','shooter','wounds','dead','terrorist',
                           'hurt','terror','police','killed','gunman','weapon','injured','attacked','bomb',
                           'bombed','attacker']
        tweet = tweet.lower()
        tweet_list = tweet.split()
        label = -1
        for word in tweet_list:
            if word in sensitive_words:
                label = 1
        # print ('sensitive_label = ',label)
        return label

    def read_file(self,file = 'train'):
        'read from file to form feature matrix'
        assert file == 'train' or file == 'test'
        if file == 'train':
            file_path = self.train_path
        else:
            file_path = self.test_path
        with open(file_path) as f:
            context = f.readlines()
            for tweet in context:
                tweet_list = tweet.split()
                label = tweet_list[0]
                if label == '__label__t':
                    label = 1
                else:
                    label = 0
                if file == 'train':
                    self.train_label.append(label)
                else:
                    self.test_label.append(label)
                tweet = tweet.strip('\n')
                locate_feature = self.locate_feature(tweet)
                # time_feature = self.time_feature(tweet)
                url_feature = self.url_feature(tweet)
                sensitive_feature = self.sensitive_words(tweet)
                sentence_vector = self.get_sentence_vector(tweet)
                feature,self.dim = self.feature_vector(file,label,sentence_vector,locate_feature,url_feature,sensitive_feature)
                self.feature_matrix(feature,label,file)


if __name__ == '__main__':
    train_path = '/home/nfs/yangl/event_detection/testdata/fast_gbdt/train.txt'
    test_path = '/home/nfs/yangl/event_detection/testdata/fast_gbdt/test.txt'
    clf = Classifier(train_path, test_path)
    clf.read_file('train')  # train model
    clf.read_file('test')  # test
    print(clf.predict_prob())


import re
import __init__
from TweetDict import TweetDict
import FileIterator

from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf


class WordFreqCounter:
    def __init__(self, twdict=None, capignore=True):
        self.doc_num = 0
        self.capignore = capignore
        self.stopworddict = self.create_stopword_dict()
        if twdict is not None:
            self.twdict = twdict
            self.reset_freq_couter()
        else:
            self.twdict = TweetDict()
    
    def word_id(self, word):
        return self.twdict.word_id(word)
    
    def is_word_in_dict(self, word):
        return self.twdict.is_word_in_dict(word)
    
    def vocabulary(self):
        return self.twdict.vocabulary()
    
    def vocabulary_size(self):
        return self.twdict.vocabulary_size()
    
    @staticmethod
    def create_stopword_dict():
        stopdict = {}
        for stopword in stopwords.words('english'):
            stopdict[stopword] = True
        return stopdict
    
    def is_valid_keyword(self, word):
        # Assume that word has been properly processed.
        isword = re.search('^[^A-Za-z]+$', word) is None
        notchar = re.search('^\w$', word) is None
        notstopword = word not in self.stopworddict
        return isword and notchar and notstopword
    
    def is_valid_wordlabel(self, wordlabel):
        notentity = wordlabel[1].startswith('O')
        return notentity
    
    def reset_ids(self):
        self.twdict.reset_ids()
    
    def reset_freq_couter(self):
        self.doc_num = 0
        for word in self.twdict.worddict:
            self.twdict.worddict[word]['df'] = 0
            self.twdict.worddict[word]['idf'] = 0
    
    def calculate_idf(self):
        if self.doc_num == 0:
            raise ValueError('No valid word has been recorded yet.')
        for word in self.twdict.worddict:
            df = self.twdict.worddict[word]['df'] + 1
            self.twdict.worddict[word]['idf'] = np.log(self.doc_num / df)
    
    def expand_dict_from_word(self, word):
        self.twdict.expand_dict_from_word(word)
    
    def expand_dict_and_count_df_from_wordlabels(self, wordlabels):
        added_word = {}
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word:
                    continue
                added_word[word] = True
                # "word" is now neither entity nor invalid keyword or duplicated word
                self.expand_dict_from_word(word)
                if 'df' not in self.twdict.worddict[word]:
                    self.twdict.worddict[word]['df'] = 1
                else:
                    self.twdict.worddict[word]['df'] += 1
        self.doc_num += 1
    
    def remove_word_by_idf_threshold(self, rmv_cond):
        self.calculate_idf()
        for word in sorted(self.twdict.worddict.keys()):
            word_idf = self.twdict.worddict[word]['idf']
            if not rmv_cond(word_idf):
                self.remove_words([word])
    
    def remove_words(self, wordarray, updateid=False):
        self.twdict.remove_words(wordarray)
        if updateid:
            self.reset_ids()
    
    def idf_vector_of_wordlabels(self, wordlabels):
        added_word = {}
        vector = np.zeros(self.vocabulary_size(), dtype=np.float32)
        for wordlabel in wordlabels:
            word = wordlabel[0].lower() if self.capignore else wordlabel[0]
            if not (self.is_valid_keyword(word) and self.is_valid_wordlabel(wordlabel)):
                continue
            else:
                if word in added_word:
                    continue
                added_word[word] = True
                wordid = self.word_id(word)
                if wordid:
                    vector[wordid] = self.twdict.worddict[word]['idf']
        return vector
    
    
    
    
    
    
    
    
    def init_params(self):
        vocab_size = self.vocabulary_size()
        self.thetaEW = tf.Variable(tf.random_normal([1, vocab_size], dtype=tf.float32))
        self.thetaEb = tf.Variable(tf.random_normal([1], dtype=tf.float32))
        
        self.seedxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.seedscore = tf.nn.xw_plus_b(self.seedxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.seedloss = tf.reduce_sum(tf.log(tf.sigmoid(self.seedscore)))
        
        self.unlbxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.unlbye = tf.placeholder(tf.float32, [None, 1])
        self.unlbscore = tf.nn.xw_plus_b(self.unlbxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.unlbpredave = tf.reduce_mean(tf.sigmoid(self.unlbscore))
        self.unlbcross = self.cross_entropy(self.unlbpredave, self.unlbye) - \
            self.cross_entropy(self.unlbye, self.unlbye)
        self.unlbloss = tf.reduce_mean(self.unlbcross)
        
        self.l2reg = tf.nn.l2_loss(self.thetaEW) + tf.nn.l2_loss(self.thetaEb)
        unlbreg_lambda = 0.2
        l2reg_lambda = 0.2
        # self.loss = self.seedloss + unlb_lambda * self.unlbloss + l2reg_lambda * self.l2reg
        self.loss = unlbreg_lambda * self.unlbloss + l2reg_lambda * self.l2reg
        self.trainop = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def cross_entropy(self, logits, labels):
        log_loss = labels * tf.log(logits) + (1 - labels) * tf.log(1-logits)
        return -log_loss
    
    def train_step(self, seedx, unlbx, unlby):
        # self.sess.run([self.trainop], feed_dict={self.seedxe: seedx, self.unlbxe: unlbx, self.unlbye: unlby})
        self.sess.run([self.trainop], feed_dict={self.unlbxe: unlbx, self.unlbye: unlby})
    
    def predict(self, inputvectors):
        return self.sess.run([self.seedscore], feed_dict={self.seedxe: inputvectors})
    
    # def seed_instance_loss(self, inputx, inputy):
    #     vocab_size = self.vocabulary_size()
    #     thetaEW = tf.Variable(tf.random_normal([1, vocab_size], dtype=tf.float32))
    #     thetaEb = tf.Variable(tf.random_normal([1], dtype=tf.float32))
    #     xe = tf.placeholder(tf.float32, [None, vocab_size])
    #     ye = tf.placeholder(tf.float32, [None, 1])
    #     score = tf.matmul(xe, tf.transpose(thetaEW)) + thetaEb
    #     losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=ye)
    #     loss = tf.reduce_mean(losses)
    #     # return loss
    #     optimizer = tf.train.AdamOptimizer(5e-2)
    #     operation = optimizer.minimize(loss)
    #
    #     sess = tf.InteractiveSession()
    #     sess.run(tf.global_variables_initializer())
    #     num_steps = 10000
    #     for step in range(num_steps):
    #         _, ll = sess.run([operation, loss], feed_dict={xe: inputx, ye: inputy})
    #     print(sess.run([thetaEW]))
    
    def dump_worddict(self, dict_file):
        FileIterator.dump_array(dict_file, [self.twdict.worddict])


# def test_train():
#     fc = FreqCounter()
#     for w in ['me', 'you', 'he', 'never', 'give', 'up', 'not', 'only', 'and', 'put', ]:
#         fc.expand_dict_from_word(w)
#     fc.reset_freq_couter()
#     fc.count_df_from_wordarray('he never give me'.split(' '))
#     fc.count_df_from_wordarray('you shall put and on yourself'.split(' '))
#     fc.count_df_from_wordarray('i should not put and on yourself'.split(' '))
#     fc.count_df_from_wordarray('have you ever been defeated'.split(' '))
#     fc.calculate_idf()
#     idfmatrix = list()
#     idfmatrix.append(fc.idf_vector_of_wordarray('he never give me'.split(' ')))
#     idfmatrix.append(fc.idf_vector_of_wordarray('you shall put and on yourself'.split(' ')))
#     idfmatrix.append(fc.idf_vector_of_wordarray('i should not put and on yourself'.split(' ')))
#     idfmatrix.append(fc.idf_vector_of_wordarray('have you ever been defeated'.split(' ')))
#     fc.supervised_loss(idfmatrix, [[0]]*len(idfmatrix))
#
#
# if __name__ == "__main__":
#     test_train()

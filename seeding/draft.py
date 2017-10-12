# from nltk.tokenize import TreebankWordTokenizer
# tokenizer = TreebankWordTokenizer()
#
# tokenizer.tokenize('I\'ve never said any #HelloWorld, but (no offense) the computer filed to recognize it.')
#
#
# from nltk.corpus import wordnet
# for synset in wordnet.synsets('hold'):
#     # print(synset.definition())
#     # synset.name()
#     # if len(synset.hypernyms()) > 0:
#     #     synset.hypernyms()[0].name()
#     # print(synset.pos())
#     for lemma in synset.lemmas():
#         print(lemma.name())
#     print('')
#
# syn = wordnet.synsets('strong')[0]
#
#
# wordnet.synset('good.n.02').lemmas()[0].antonyms()[0]
#
#
# from nltk.corpus import webtext
# words = [w.lower() for w in webtext.words('singles.txt')]
# from nltk.corpus import stopwords
# stopset = set(stopwords.words('english'))
# filter_stops = lambda w: len(w) < 3 or w in stopset
# from nltk.collocations import TrigramCollocationFinder
# tcf = TrigramCollocationFinder.from_words(words)
#
#
#
# from nltk.stem import PorterStemmer
# # from nltk.stem import LancasterStemmer  # More aggressive
# stemmer = PorterStemmer()
# stemmer.stem('unrestricted')
# stemmer.stem('underestimated')
# stemmer.stem('adversary')
# stemmer.stem('ain\'t')
#
#
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# lemmatizer.lemmatize('cooking')
# lemmatizer.lemmatize('underestimated')
# lemmatizer.lemmatize('adversary')
#
# lemmatizer.lemmatize(stemmer.stem('underestimated'))
#
#
# t = ['usingvariousmolecularbiology', 'techniques', 'toproduce', 'genotypes', 'following',
#      'standardoperatingprocedures', '.', '#Operateandmaintainautomatedequipment', '.',
#      '#Updatesampletrackingsystemsandprocess', 'documentation', 'toallowaccurate',
#      'monitoring', 'andrapid', 'progression', 'ofcasework']
# from wordsegment import segment
# segment(' '.join(t))
#
#
# from nltk.corpus import wordnet
# wordnet.synsets('good', pos='a')[0].lemmas()[0].antonyms()
#
#
#
# # import re
# # s = 'wp[sp;\'q+wo[di=-'
# # p=re.compile(r'(\W)*#(\W)+')
# # p.sub(' ', s)
#
#
# re.split('[\|\[\]\(\),.]', '(qwe)poi.tyu,098|')
# re.split('[\|\[\]\(\),.\^%\$\!`~\+=_\-\s]', '* Documentation:  https://help.ubuntu.com')
#
#
# from wordsegment import segment
# segment('youngbloods')
# segment('I\'m')
#
#
from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))
#
#
# from TweetDict import TweetDict
# TweetDict().contraction_patterns.apply_pattern('I\'m not going down'.lower())
#
#
# s='2016<<_11_10_20.sum'
# dot_idx = s.rfind('.')
# s[dot_idx-13: dot_idx]
#
# from Pattern import get_pattern
# get_pattern().full_split_nondigit('2016<<_11_10_20.sum')
#
#
# def remo(arr):
#     del arr[2]
#     return arr
#
# a = [1,2,3,4,5]
# remo(a)
#
# '2123asd.sum'[:-4]
#
#
# def parse_ner_word_into_labels(ner_word, slash_num):
#     res = []
#     over = False
#     for i in range(slash_num):
#         idx = ner_word.rfind('/') + 1
#         res.insert(0, ner_word[idx:])
#         ner_word = ner_word[0:idx - 1]
#         if idx == 0:
#             over = True
#             break
#     if not over:
#         res.insert(0, ner_word)
#     return res
#
# parse_ner_word_into_labels('wqe', slash_num=2)
#
#
# import re
# from JsonParser import JsonParser
# j=JsonParser()
# text = 'RT @Clue_BBP: new month\nnew vibes\nnew money\nnew opportunities. '
# text='RT @The_ashima: We bow to the culture and simplicity of the state and its people! Congratulations to everyone. :) @Gurmeetramrahim Ji'
# text='RT @NPP_GH: #WeSeeYou and we see CORRUPTION, INCOMPETENCE, MISMANAGEMENT, SUFFERING.'
# normalized_text = j.pattern.normalization(text)
# print(normalized_text)
#
# 793339741898993664
#
# '%^sdq()'.strip()
#
#
# def func(f):
#     x=3
#     return f(x)
#
# func(lambda idf: idf > 3)
#
# import random
# for i in range(10000):
#     if random.random() < 1/1000:
#         print(1)
#
#
# from SeedQuery import SeedQuery
#
# since = ['2016', '11', '1']
# until = ['2016', '11', '5']
# s = SeedQuery({'all_of': ['terror']}, since, until)
# s.is_text_desired('Dog pulled from rubble after 6 . 6 magnitude earthquake in Italy')
#
#
# import tensorflow as tf
# def cross_entropy(logits, labels):
#     log_loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)
#     return - log_loss
#
# e = tf.placeholder(tf.float32, [1, None])
# f = tf.placeholder(tf.float32, [1, None])
# c = cross_entropy(logits=e, labels=f)
#
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# sess.run([c], feed_dict={e:[[0.1,0.2,0.3,0.5]], f:[[0.5]]})
#
#
# from operator import itemgetter
# idf_ranked_list = sorted({1:2, 2:10, 4:3}.items(), key=itemgetter(1))
# for idx, (a, b) in enumerate(idf_ranked_list):
#     print(idx, a, b)
#
#
#
# import tag_converter
# from tag_converter import Tagger
# t=Tagger()
# t.load_untagged('/home/nfs/cdong/tw/seeding/Terrorist/Terrorist.utg')
#
#
#
#
# a=[6,4,7,9,4]
# for i in a:
#     if i < 7:
#         a.pop(i)
#


# import sys
#
# while True:
#     sys.stdout.write('name: ')
#     sys.stdout.flush()
#     print(sys.stdin.readline().strip()+'-')


import sys


def handpick_words_of_theme(in_dict, theme):
    print('Judge if the word can imply the theme of:', theme)
    print('"1" indicates YES ,while "2" means NO; "3" to go back to the previous word')
    tobe_removed = {}
    words = sorted(list(in_dict))
    i = 0
    while i < len(words):
        key = words[i]
        while True:
            sys.stdout.write(' '.join(['word:', '"'+key+'"', ', judge:']))
            sys.stdout.flush()
            judge = sys.stdin.readline().strip()
            if '1' in judge:
                tobe_removed[key] = False
                print('"'+key+'"', 'reserved')
                i += 1
            elif '2' in judge:
                tobe_removed[key] = True
                print('"'+key+'"', 'popped')
                i += 1
            elif "3" in judge:
                print('')
                i -= (1 if i > 0 else 0)
            else:
                print('Re-enter your judge.')
                continue
            break
    for word in tobe_removed.keys():
        if tobe_removed[word]:
            in_dict.pop(word)


# d = {'my': 12, 'no': 2, 'kill': 9, 'damn': 53, 'shoot': 0, 'hate': 0, 'fire': 0, 'bomb': 0, 'cake': 0, 'movie': 0,
#      'attack': 0, 'football': 0, 'swim': 0, 'gold': 0, 'money': 0, 'can': 0, 'make': 0, 'death': 0, 'explode': 0, }
# d = {'my': 12, 'no': 2, 'kill': 9, 'damn': 53, }
# handpick_words_of_theme(d, 'Terrorist')
# print(d)


# 默认模式r,读
import zipfile
azip = zipfile.ZipFile('src.zip')  # ['bb/', 'bb/aa.txt']
print(azip.namelist())






# Cho, K., Courville, A., Bengio, Y., 2015. Describing multimedia
# content using attention-based encoder-decoder
# networks. IEEE Trans. Multim., 17(11):1875-1886.
# http://dx.doi.org/10.1109/TMM.2015.2477044
#
#
# Kalchbrenner, N., Grefenstette, E., Blunsom, P., 2014. A
# convolutional neural network for modelling sentences.
# ePrint Archive, arXiv:1404.2188.
#
# # Li, J.W., Monroe, W., Ritter, A., et al., 2016. Deep reinforcement
# # learning for dialogue generation. ePrint
# # Archive, arXiv:1606.01541.
#
# Liu, Y., Sun, C.J., Lin, L., et al., 2016. Learning natural
# language inference using bidirectional LSTM model and
# inner-attention. ePrint Archive, arXiv:1605.09090.
#
# Low, Y.C., Gonzalez, J.E., Kyrola, A., et al., 2014.
# GraphLab: a new framework for parallel machine learning.
# ePrint Archive, arXiv:1408.2041.
#
#
# Marrinan, T., Aurisano, J., Nishimoto, A., et al., 2014.
# SAGE2: a new approach for data intensive collaboration
# using scalable resolution shared displays. Int.
# Conf. on Collaborative Computing: Networking,
# Applications and Worksharing (CollaborateCom),
# p.177-186.
#
# Mikolov, T., Chen, K., Corrado, G., et al., 2013. Efficient estimation
# of word representations in vector space. ePrint
# Archive, arXiv:1301.3781.
#
# Shijia, E., Jia, S.B., Yang, X., et al., 2016. Knowledge graph
# embedding for link prediction and triplet classification.
# China Conf. on Knowledge Graph and Semantic Computing:
# Semantic, Knowledge, and Linked Big Data,
# p.228-232.
# http://dx.doi.org/10.1007/978-981-10-3168-7_23
#
# Sutskever, I., Vinyals, O., Le, Q.V., 2014. Sequence to
# sequence learning with neural networks. Conf. on
# Neural Information Processing Systems, p.3104-3112.
#
# Weston, J., Chopra, S., Bordes, A., 2014. Memory networks.
# ePrint Archive, arXiv:1410.3916.
#
# Wu, F., Yu, Z., Yang, Y., et al., 2014. Sparse multi-modal
# hashing. IEEE Trans. Multim., 16(2):427-439.
# http://dx.doi.org/10.1109/TMM.2013.2291214
#
# Wu, F., Jiang, X.Y., Li, X., et al., 2015. Cross-modal
# learning to rank via latent joint representation. IEEE
# Trans. Imag. Process., 24(5):1497-1509.
# http://dx.doi.org/10.1109/TIP.2015.2403240

import re
def remove_noneword_from_wordlabels(wordlabels):
    for idx, wordlabel in enumerate(wordlabels):
        if re.search('^[^a-zA-Z0-9]+$', wordlabel) is not None:
            print('pre', wordlabels)
            print(idx)
            del wordlabels[idx]
            print('post', wordlabels)
    return wordlabels

w = ['i', '\'', 'm', ' ', 'not', 'going', '&', ]
remove_noneword_from_wordlabels(w)



from EventClassifier import EventClassifier
ec = EventClassifier(10, 1e-3)
print(ec.get_theta())
ec.reserve_params('/home/nfs/cdong/tw/seeding/params.p')
ec.restore_params('/home/nfs/cdong/tw/seeding/params.p')
print(ec.get_theta())



class a:
    def __init__(self, age):
        self.age = age

class b:
    def __init__(self):
        self.a = a(-1)

from copy import deepcopy, copy
b1 = b()
b2 = deepcopy(b())
b3 = copy(b1)


b1.a.age = 0
b2.a.age = 233
b3.a.age = 999

b1.a.age
b2.a.age
b3.a.age

from EventFeatureExtractor import EventFeatureExtractor
e = EventFeatureExtractor()
e.__name__


class AA:
    def __init__(self, age):
        self.age = age

a = AA(233)
b=a.__class__(555)
b.age


import multiprocessing as mp
def f(a, b):
    return a+b


d
parma = [(1,2), (3,4), (5,6), (6,7), (9,1), (2,4), (5,2), (3,7)]


process_num = 8
pool = mp.Pool(processes=process_num)
res_getter = []
for i in range(process_num):
    res = pool.apply_async(func=f, args=parma[i])
    res_getter.append(res)

pool.close()
pool.join()
results = []
for i in range(process_num):
    results.append(res_getter[i].get())

results






import tensorflow as tf
import numpy as np
from EventClassifier import EventClassifier

ec = EventClassifier(100, 0)
ec.restore_params('./what.prams')
print(ec.predict(np.random.random([1,100])))

ec2 = EventClassifier(1, 0)
ec2.restore_params('./what.prams')
print(ec2.predict(np.random.random([1,100])))



import Levenshtein
import __init__
import TweetKeys
def remove_similar_tws(twarr, sim_threashold=0):
    tw_id_cpy = sorted([(idx, tw) for idx, tw in enumerate(twarr)], key=lambda item: item[1][TweetKeys.key_cleantext])
    prev = 0
    remove_ids = list()
    for cur in range(1, len(tw_id_cpy)):
        if Levenshtein.distance(tw_id_cpy[prev][1][TweetKeys.key_cleantext],
                                tw_id_cpy[cur][1][TweetKeys.key_cleantext]) < 5:
            remove_ids.append(tw_id_cpy[cur][0])
        else:
            prev = cur
    for idx in sorted(remove_ids, reverse=True):
        del twarr[idx]
    return twarr


# remove_similar_tws([{'text':'asdfg'}, {'text':'asdfg'}, {'text':'asdfg'}, {'text':'asdfg'},{'text':'wueigf'}, ])

import Levenshtein
Levenshtein.distance("Burns to the bone White phosphorus used in #Mosul may cause civilian suffering #Amnesty Int'l",
                     "RT com Burns to the bone White phosphorus used in #Mosul may cause civilian suffering #Amnesty Int'l")






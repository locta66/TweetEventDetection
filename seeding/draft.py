from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

tokenizer.tokenize('I\'ve never said any #HelloWorld, but (no offense) the computer filed to recognize it.')


from nltk.corpus import wordnet
for synset in wordnet.synsets('hold'):
    # print(synset.definition())
    # synset.name()
    # if len(synset.hypernyms()) > 0:
    #     synset.hypernyms()[0].name()
    # print(synset.pos())
    for lemma in synset.lemmas():
        print(lemma.name())
    print('')

syn = wordnet.synsets('strong')[0]


wordnet.synset('good.n.02').lemmas()[0].antonyms()[0]


from nltk.corpus import webtext
words = [w.lower() for w in webtext.words('singles.txt')]
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset
from nltk.collocations import TrigramCollocationFinder
tcf = TrigramCollocationFinder.from_words(words)



from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer  # More aggressive
stemmer = PorterStemmer()
stemmer.stem('unrestricted')
stemmer.stem('underestimated')
stemmer.stem('adversary')
stemmer.stem('ain\'t')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('cooking')
lemmatizer.lemmatize('underestimated')
lemmatizer.lemmatize('adversary')

lemmatizer.lemmatize(stemmer.stem('underestimated'))


t = ['usingvariousmolecularbiology', 'techniques', 'toproduce', 'genotypes', 'following',
     'standardoperatingprocedures', '.', '#Operateandmaintainautomatedequipment', '.',
     '#Updatesampletrackingsystemsandprocess', 'documentation', 'toallowaccurate',
     'monitoring', 'andrapid', 'progression', 'ofcasework']
from wordsegment import segment
segment(' '.join(t))


from nltk.corpus import wordnet
wordnet.synsets('good', pos='a')[0].lemmas()[0].antonyms()



# import re
# s = 'wp[sp;\'q+wo[di=-'
# p=re.compile(r'(\W)*#(\W)+')
# p.sub(' ', s)


re.split('[\|\[\]\(\),.]', '(qwe)poi.tyu,098|')
re.split('[\|\[\]\(\),.\^%\$\!`~\+=_\-\s]', '* Documentation:  https://help.ubuntu.com')


from wordsegment import segment
segment('youngbloods')
segment('I\'m')


from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))


from TweetDict import TweetDict
TweetDict().contraction_patterns.apply_pattern('I\'m not going down'.lower())


s='2016<<_11_10_20.sum'
dot_idx = s.rfind('.')
s[dot_idx-13: dot_idx]

from Pattern import get_pattern
get_pattern().full_split_nondigit('2016<<_11_10_20.sum')


def remo(arr):
    del arr[2]
    return arr

a = [1,2,3,4,5]
remo(a)

'2123asd.sum'[:-4]


def parse_ner_word_into_labels(ner_word, slash_num):
    res = []
    over = False
    for i in range(slash_num):
        idx = ner_word.rfind('/') + 1
        res.insert(0, ner_word[idx:])
        ner_word = ner_word[0:idx - 1]
        if idx == 0:
            over = True
            break
    if not over:
        res.insert(0, ner_word)
    return res

parse_ner_word_into_labels('wqe', slash_num=2)


import re
from JsonParser import JsonParser
j=JsonParser()
text = 'RT @Clue_BBP: new month\nnew vibes\nnew money\nnew opportunities. '
text='RT @The_ashima: We bow to the culture and simplicity of the state and its people! Congratulations to everyone. :) @Gurmeetramrahim Ji'
text='RT @NPP_GH: #WeSeeYou and we see CORRUPTION, INCOMPETENCE, MISMANAGEMENT, SUFFERING.'
normalized_text = j.pattern.normalization(text)
print(normalized_text)

793339741898993664

'%^sdq()'.strip()


def func(f):
    x=3
    return f(x)

func(lambda idf: idf > 3)

import random
for i in range(10000):
    if random.random() < 1/1000:
        print(1)


from SeedQuery import SeedQuery

since = ['2016', '11', '1']
until = ['2016', '11', '5']
s = SeedQuery({'all_of': ['terror']}, since, until)
s.is_text_desired('Dog pulled from rubble after 6 . 6 magnitude earthquake in Italy')


import tensorflow as tf
def cross_entropy(logits, labels):
    log_loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)
    return - log_loss

e = tf.placeholder(tf.float32, [1, None])
f = tf.placeholder(tf.float32, [1, None])
c = cross_entropy(logits=e, labels=f)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run([c], feed_dict={e:[[0.1,0.2,0.3,0.5]], f:[[0.5]]})


from operator import itemgetter
idf_ranked_list = sorted({1:2, 2:10, 4:3}.items(), key=itemgetter(1))
for idx, (a, b) in enumerate(idf_ranked_list):
    print(idx, a, b)



import tag_converter
from tag_converter import Tagger
t=Tagger()
t.load_untagged('/home/nfs/cdong/tw/seeding/Terrorist/Terrorist.utg')




a=[6,4,7,9,4]
for i in a:
    if i < 7:
        a.pop(i)


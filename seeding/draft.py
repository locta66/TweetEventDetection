# from nltk.tokenize import TreebankWordTokenizer
# tokenizer = TreebankWordTokenizer()
# tokenizer.tokenize('I\'ve never said any #HelloWorld, but (no offense) the computer filed to recognize it.')

from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))
len(english_stops)
import nltk
nltk.help.upenn_tagset()


# import re
# def remove_noneword_from_wordlabels(wordlabels):
#     for idx, wordlabel in enumerate(wordlabels):
#         if re.search('^[^a-zA-Z0-9]+$', wordlabel) is not None:
#             print('pre', wordlabels)
#             print(idx)
#             del wordlabels[idx]
#             print('post', wordlabels)
#     return wordlabels
#
# w = ['i', '\'', 'm', ' ', 'not', 'going', '&', ]
# remove_noneword_from_wordlabels(w)

import tensorflow as tf

vocab_size = 6
thetaEW = tf.Variable([[1, 2, 3]], dtype=tf.float32)
# thetaEb = tf.Variable(tf.ones([1], dtype=tf.float32))
#
# seedxe = tf.placeholder(tf.float32, [None, vocab_size])
# seedye = tf.placeholder(tf.float32, [None, 1])
# seedscore = tf.nn.xw_plus_b(seedxe, tf.transpose(thetaEW), thetaEb)
# seedpred = tf.sigmoid(seedscore)
# seedloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=seedscore, labels=seedye)
#
# idfmtx = [[1, 1, 1], [0, 0, 1], ]
# lblmtx = [[1], [0], ]
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# sess.run([seedpred, seedloss], feed_dict={seedxe: idfmtx, seedye: lblmtx})
sess.run([thetaEW, tf.nn.l2_loss(thetaEW)])


# a=[[1,2,3], [11,22,33], [111,222,333], ]
# aa=np.array(a)
# aa.put(indices=[-1], values=[[3,4,5]])

print(len([('#', 'B-NP'), ('$', 'B-NP'), ("''", 'O'), ('(', 'O'), (')', 'O'),
           (',', 'O'), ('.', 'O'), (':', 'O'), ('CC', 'O'), ('CD', 'I-NP'),
           ('DT', 'B-NP'), ('EX', 'B-NP'), ('FW', 'I-NP'), ('IN', 'O'),
           ('JJ', 'I-NP'), ('JJR', 'B-NP'), ('JJS', 'I-NP'), ('MD', 'O'),
           ('NN', 'I-NP'), ('NNP', 'I-NP'), ('NNPS', 'I-NP'), ('NNS', 'I-NP'),
           ('PDT', 'B-NP'), ('POS', 'B-NP'), ('PRP', 'B-NP'), ('PRP$', 'B-NP'),
           ('RB', 'O'), ('RBR', 'O'), ('RBS', 'B-NP'), ('RP', 'O'), ('SYM', 'O'),
           ('TO', 'O'), ('UH', 'O'), ('VB', 'O'), ('VBD', 'O'), ('VBG', 'O'),
           ('VBN', 'O'), ('VBP', 'O'), ('VBZ', 'O'), ('WDT', 'B-NP'),
           ('WP', 'B-NP'), ('WP$', 'B-NP'), ('WRB', 'O'), ('``', 'O')]))

import numpy as np
X = np.matrix([[0.65, 0.35, 0.0],
               [0.15, 0.67, 0.18],
               [0.12, 0.36, 0.52]])
for _ in range(10):
    X = X * X
    print(X, '\n')


import pandas as pd
d = pd.DataFrame({'pred': [2,2,3,4,4,4,4,5,5,6,7,7,7], 'label': [1,2,1,5,5,4,3,1,1,2,3,3,4]})
d.groupby('pred').groups

label = [11,2,1,55,55,4,3,1,11,2,3,33,4]
dic = d.groupby('pred').indices
print(type(dic))
for pred in dic:
    dic[pred] = [label[i] for i in dic[pred]]


import pandas as pd
pred = [9,8,7,6,1]
label = [11,2,1,55,55,4,3,1,11,2,3,33,4]
df = pd.DataFrame(index=sorted(pred), columns=set(label), data=0)
# df.append({11:1, 4:2}, ignore_index=True, verify_integrity=False)
# df.fillna(0)


import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [5, 7, 4], '^-', lw=1, label='cosine', color='r')
plt.legend(loc='lower right')
# plt.grid(True, linestyle="-", color="#333333", linewidth="0.1")
plt.text(2.75, 7.3, 'matplotlib', verticalalignment="bottom",horizontalalignment="left")
plt.show()
plt.savefig("examples.png")


# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(x,y,'k--')
# ax.set_xticks([0,25,50,75,100])
# ax.set_xticklabels(['one','two','three','four','five'],rotation=45,fontsize=    'small')
# ax.set_title('Demo Figure')
# ax.set_xlabel('Time')
# plt.show()


import numpy as np
from scipy import sparse, io
mtx = [[1.2, 2.3], [4.2, 3.1]]*30000
contract_mtx = sparse.csr_matrix(mtx)
io.mmwrite('shit.sss', contract_mtx)
read_contract_mtx = io.mmread('shit.sss')
dense_mtx = read_contract_mtx.todense()
print(np.sum(np.matrix(mtx) - np.matrix(dense_mtx)))

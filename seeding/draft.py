# # from nltk.tokenize import TreebankWordTokenizer
# # tokenizer = TreebankWordTokenizer()
# # tokenizer.tokenize('I\'ve never said any #HelloWorld, but (no offense) the computer filed to recognize it.')
# #
# #
# # t = ['usingvariousmolecularbiology', 'techniques', 'toproduce', 'genotypes', 'following',
# #      'standardoperatingprocedures', '.', '#Operateandmaintainautomatedequipment', '.',
# #      '#Updatesampletrackingsystemsandprocess', 'documentation', 'toallowaccurate',
# #      'monitoring', 'andrapid', 'progression', 'ofcasework']
# # from wordsegment import segment
# # segment(' '.join(t))
# #
# #
# from nltk.corpus import stopwords
# english_stops = set(stopwords.words('english'))
# len(english_stops)
# #
# #
#
#
# # Cho, K., Courville, A., Bengio, Y., 2015. Describing multimedia
# # content using attention-based encoder-decoder
# # networks. IEEE Trans. Multim., 17(11):1875-1886.
# # http://dx.doi.org/10.1109/TMM.2015.2477044
# #
# #
# # Kalchbrenner, N., Grefenstette, E., Blunsom, P., 2014. A
# # convolutional neural network for modelling sentences.
# # ePrint Archive, arXiv:1404.2188.
# #
# # # Li, J.W., Monroe, W., Ritter, A., et al., 2016. Deep reinforcement
# # # learning for dialogue generation. ePrint
# # # Archive, arXiv:1606.01541.
# #
# # Liu, Y., Sun, C.J., Lin, L., et al., 2016. Learning natural
# # language inference using bidirectional LSTM model and
# # inner-attention. ePrint Archive, arXiv:1605.09090.
# #
# # Low, Y.C., Gonzalez, J.E., Kyrola, A., et al., 2014.
# # GraphLab: a new framework for parallel machine learning.
# # ePrint Archive, arXiv:1408.2041.
# #
# #
# # Marrinan, T., Aurisano, J., Nishimoto, A., et al., 2014.
# # SAGE2: a new approach for data intensive collaboration
# # using scalable resolution shared displays. Int.
# # Conf. on Collaborative Computing: Networking,
# # Applications and Worksharing (CollaborateCom),
# # p.177-186.
# #
# # Mikolov, T., Chen, K., Corrado, G., et al., 2013. Efficient estimation
# # of word representations in vector space. ePrint
# # Archive, arXiv:1301.3781.
# #
# # Shijia, E., Jia, S.B., Yang, X., et al., 2016. Knowledge graph
# # embedding for link prediction and triplet classification.
# # China Conf. on Knowledge Graph and Semantic Computing:
# # Semantic, Knowledge, and Linked Big Data,
# # p.228-232.
# # http://dx.doi.org/10.1007/978-981-10-3168-7_23
# #
# # Sutskever, I., Vinyals, O., Le, Q.V., 2014. Sequence to
# # sequence learning with neural networks. Conf. on
# # Neural Information Processing Systems, p.3104-3112.
# #
# # Weston, J., Chopra, S., Bordes, A., 2014. Memory networks.
# # ePrint Archive, arXiv:1410.3916.
# #
# # Wu, F., Yu, Z., Yang, Y., et al., 2014. Sparse multi-modal
# # hashing. IEEE Trans. Multim., 16(2):427-439.
# # http://dx.doi.org/10.1109/TMM.2013.2291214
# #
# # Wu, F., Jiang, X.Y., Li, X., et al., 2015. Cross-modal
# # learning to rank via latent joint representation. IEEE
# # Trans. Imag. Process., 24(5):1497-1509.
# # http://dx.doi.org/10.1109/TIP.2015.2403240
#
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
#
#
#
# from EventClassifier import EventClassifier
# ec = EventClassifier(10, 1e-3)
# print(ec.get_theta())
# ec.reserve_params('/home/nfs/cdong/tw/seeding/params.p')
# ec.restore_params('/home/nfs/cdong/tw/seeding/params.p')
# print(ec.get_theta())




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
#
#
# import __init__
# import ArrayUtils
# import numpy as np

# a = np.zeros([5, 1], dtype=np.int)
# b = np.ones([4, 1], dtype=np.int)
# x = np.random.random([5, 3])
# x
# y = np.random.random([4, 3])
# y
#
# ArrayUtils.arrays_partition([np.concatenate((a, b), axis=0), np.concatenate((x, y), axis=0)], partition_arr=(4, 4))


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

import nltk
nltk.help.upenn_tagset()

import numpy as np
from sklearn import metrics

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
pairs = [(y_scores[i], y_true[i]) for i in range(len(y_true))]

# metrics.roc_auc_score(y_true, y_scores)
precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores,
                                                               sample_weight=[0.1, 0.2, 0.4, 0.9])
span = int(len(y_true) / 2)
print([precision[i * span] for i in range(2)])
print([recall[i * span] for i in range(2)])

# fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
# metrics.auc(fpr, tpr, reorder=True)
# ArrayUtils.roc(pairs)
# ArrayUtils.recall(pairs, [i / 10 for i in range(1, 10)])
# ArrayUtils.precision(pairs, [i / 10 for i in range(1, 10)])

# import multiprocessing
# print(multiprocessing.cpu_count())


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

import numpy as np
a = np.array([11,22,11,22,33,33,22,11,44])
counts = np.bincount(a)
np.argmax(counts)

L=[1,2,3,2,4,33]
from collections import Counter
counter = Counter(L)
list(counter)


from collections import Counter
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
# plt.savefig("examples.png")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = [-1, 0]
Y = [-2, -1]
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(x,y,'k--')
# ax.set_xticks([0,25,50,75,100])
# ax.set_xticklabels(['one','two','three','four','five'],rotation=45,fontsize=    'small')
# ax.set_title('Demo Figure')
# ax.set_xlabel('Time')
# plt.show()

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

X = [1.,2.,3.,4.,5.]
Y = [3.,4.,5.,6.,7.]

nmi_frame = pd.DataFrame(index=X, columns=Y, data=0.1)
nmi_frame.loc[1][3] = 0.2
nmi_frame.loc[1][4] = 0.3
nmi_frame.loc[1][5] = 0.4
nmi_frame.loc[2][3] = 0.5
nmi_frame.loc[2][5] = 0.6
nmi_frame.loc[3][3] = 0.7
nmi_frame.loc[3][4] = 0.8
nmi_frame.loc[3][5] = 0.9

fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(X, Y)
# ax.plot_surface([1,2,3], [3,4,5], np.array(nmi_frame).tolist(), rstride=1, cstride=1, cmap='rainbow')
ax.plot_surface(X, Y, np.array(nmi_frame.T).tolist(), vmin=0, cmap='BrBG_r')
fig.savefig('e.png')

import multiprocessing as mp
mp.cpu_count()



import numpy as np
from scipy import sparse, io
mtx = [[1.2, 2.3], [4.2, 3.1]]*30000
contract_mtx = sparse.csr_matrix(mtx)
io.mmwrite('shit.sss', contract_mtx)
read_contract_mtx = io.mmread('shit.sss')
dense_mtx = read_contract_mtx.todense()
print(np.sum(np.matrix(mtx) - np.matrix(dense_mtx)))


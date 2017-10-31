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
# import sys
# def handpick_words_of_theme(in_dict, theme):
#     print('Judge if the word can imply the theme of:', theme)
#     print('"1" indicates YES ,while "2" means NO; "3" to go back to the previous word')
#     tobe_removed = {}
#     words = sorted(list(in_dict))
#     i = 0
#     while i < len(words):
#         key = words[i]
#         while True:
#             sys.stdout.write(' '.join(['word:', '"'+key+'"', ', judge:']))
#             sys.stdout.flush()
#             judge = sys.stdin.readline().strip()
#             if '1' in judge:
#                 tobe_removed[key] = False
#                 print('"'+key+'"', 'reserved')
#                 i += 1
#             elif '2' in judge:
#                 tobe_removed[key] = True
#                 print('"'+key+'"', 'popped')
#                 i += 1
#             elif "3" in judge:
#                 print('')
#                 i -= (1 if i > 0 else 0)
#             else:
#                 print('Re-enter your judge.')
#                 continue
#             break
#     for word in tobe_removed.keys():
#         if tobe_removed[word]:
#             in_dict.pop(word)
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
#
#
#
#
#
# import tensorflow as tf
# import numpy as np
# from EventClassifier import EventClassifier
#
# ec = EventClassifier(100, 0)
# ec.restore_params('./what.prams')
# print(ec.predict(np.random.random([1,100])))
#
# ec2 = EventClassifier(1, 0)
# ec2.restore_params('./what.prams')
# print(ec2.predict(np.random.random([1,100])))
#
#
# import __init__
# import time
# import FileIterator
# s = time.time()
# twarr = FileIterator.load_array('/home/nfs/cdong/tw/summary/2016_07_01_14.sum')
# print('time elapsed:', time.time() - s, 's')




# import tensorflow as tf
#
# vocab_size = 3
# thetaEW = tf.Variable(tf.ones([1, vocab_size], dtype=tf.float32))
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
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# sess.run([seedpred, seedloss], feed_dict={seedxe: idfmtx, seedye: lblmtx})
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


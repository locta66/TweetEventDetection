import tensorflow as tf


class EventClassifier:
    def __init__(self, vocab_size, learning_rate, unlbreg_lambda=0.2, l2reg_lambda=0.2):
        self.thetaEW = tf.Variable(tf.random_normal([1, vocab_size], dtype=tf.float32), name='event_thetaEW')
        self.thetaEb = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='event_thetaEb')
        self.params = [self.thetaEW, self.thetaEb]
        
        self.seedxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.seedscore = tf.nn.xw_plus_b(self.seedxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.seedpred = tf.sigmoid(self.seedscore)
        self.seedloss = -tf.reduce_sum(tf.log(self.seedpred))
        
        self.unlbxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.unlbye = tf.placeholder(tf.float32, [None, 1])
        self.unlbscore = tf.nn.xw_plus_b(self.unlbxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.unlbpred = tf.sigmoid(self.unlbscore)
        self.unlbpredave = tf.reduce_mean(self.unlbpred)
        self.unlbcross = self.cross_entropy(self.unlbpredave, self.unlbye) - \
            self.cross_entropy(self.unlbye, self.unlbye)
        # self.unlbcross = self.cross_entropy(self.unlbpredave, self.unlbye)
        self.unlbloss = tf.reduce_mean(self.unlbcross)
        
        self.unlbreg_lambda = unlbreg_lambda
        self.l2reg_lambda = l2reg_lambda
        self.l2reg = tf.nn.l2_loss(self.thetaEW) + tf.nn.l2_loss(self.thetaEb)
        # self.loss = self.seedloss + unlb_lambda * self.unlbloss + l2reg_lambda * self.l2reg
        self.loss = self.unlbreg_lambda * self.unlbloss + self.l2reg_lambda * self.l2reg
        self.trainop = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def cross_entropy(self, logits, labels):
        log_loss = labels * tf.log(logits) + (tf.constant(1, dtype=tf.float32) - labels) * tf.log(1 - logits)
        return - log_loss
    
    def train_steps(self, stepnum, seedx, unlbx, unlby):
        loss = 0
        for i in range(stepnum):
            loss = self.train_per_step(seedx, unlbx, unlby)
            print(i, 'th ,loss', loss)
        return loss
    
    def train_per_step(self, seedx, unlbx, unlby):
        # self.sess.run([self.trainop], feed_dict={self.seedxe: seedx, self.unlbxe: unlbx, self.unlbye: unlby})
        _, loss = self.sess.run([self.trainop, self.loss], feed_dict={self.unlbxe: unlbx, self.unlbye: unlby})
        return loss
    
    def predict(self, idfmtx):
        return self.sess.run([self.seedpred, self.seedloss], feed_dict={self.seedxe: idfmtx})
    
    def unlabel_predict(self, idfmtx, label):
        return self.sess.run([self.unlbpred, self.unlbpredave], feed_dict={self.unlbxe: idfmtx, self.unlbye: label})
    
    def get_theta(self):
        return self.sess.run(self.params)
    
    def reserve_params(self, file_name):
        saver = tf.train.Saver(self.params)
        saver.save(self.sess, file_name)
    
    def restore_params(self, file_name):
        saver = tf.train.Saver(self.params)
        saver.save(self.sess, file_name)

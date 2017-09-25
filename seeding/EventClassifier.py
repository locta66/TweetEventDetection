import tensorflow as tf


class EventClassifier:
    def __init__(self, vocab_size, learning_rate):
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
        self.trainop = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def cross_entropy(self, logits, labels):
        log_loss = labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits)
        return -log_loss
    
    def train_steps(self, stepnum, threshold, seedx, unlbx, unlby):
        loss = 0.0
        prev_loss = 1.0
        for i in range(stepnum):
            if prev_loss - loss < threshold:
                break
            loss = self.train_per_step(seedx, unlbx, unlby)
        return loss
    
    def train_per_step(self, seedx, unlbx, unlby):
        # self.sess.run([self.trainop], feed_dict={self.seedxe: seedx, self.unlbxe: unlbx, self.unlbye: unlby})
        _, loss = self.sess.run([self.trainop, self.loss], feed_dict={self.unlbxe: unlbx, self.unlbye: unlby})
        return loss
    
    def predict(self, inputvectors):
        return self.sess.run([self.seedscore, self.seedloss], feed_dict={self.seedxe: inputvectors})

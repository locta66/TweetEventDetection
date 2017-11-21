import tensorflow as tf


class EventClassifier:
    def __init__(self, vocab_size, learning_rate, unlbreg_lambda=0.1, l2reg_lambda=0.1):
        self.construct_calculate_graph(vocab_size, learning_rate, unlbreg_lambda, l2reg_lambda)
    
    def construct_calculate_graph(self, vocab_size, learning_rate, unlbreg_lambda, l2reg_lambda):
        self.thetaEW = tf.Variable(tf.random_normal([1, vocab_size], mean=0.3, stddev=0.3, dtype=tf.float32),
                                   name='event_thetaEW')
        # self.thetaEW = tf.Variable(tf.random_normal([1, vocab_size], dtype=tf.float32), name='event_thetaEW')
        self.thetaEb = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='event_thetaEb')
        self.params = [self.thetaEW, self.thetaEb]
        
        self.seedxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.seedye = tf.placeholder(tf.float32, [None, 1])
        self.seedscore = tf.nn.xw_plus_b(self.seedxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.seedpred = tf.sigmoid(self.seedscore)
        self.seedcross = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.seedscore, labels=self.seedye)
        self.seedloss = tf.reduce_mean(self.seedcross)
        
        self.unlbxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.unlbye = tf.placeholder(tf.float32, [1, 1])
        self.unlbscore = tf.nn.xw_plus_b(self.unlbxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.unlbpred = tf.sigmoid(self.unlbscore)
        self.unlbpredave = tf.reduce_mean(self.unlbpred)
        self.unlbcross = self.cross_entropy(self.unlbpredave, self.unlbye) - \
                         self.cross_entropy(self.unlbye, self.unlbye)
        self.unlbloss = self.unlbcross
        
        self.unlbreg_lambda = unlbreg_lambda
        self.l2reg_lambda = l2reg_lambda
        self.l2reg = tf.nn.l2_loss(self.thetaEW) + tf.nn.l2_loss(self.thetaEb)
        # self.loss = self.seedloss + self.l2reg_lambda * self.l2reg
        self.loss = self.seedloss + self.unlbreg_lambda * self.unlbloss + self.l2reg_lambda * self.l2reg
        self.trainop = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def cross_entropy(self, logits, labels):
        """tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=b)
           makes log(sigmoid(a)), and then cross it with its label b"""
        log_loss = labels * tf.log(logits) + (tf.constant(1, dtype=tf.float32) - labels) * tf.log(1 - logits)
        return - log_loss
    
    def train_steps(self, stepnum, seedx, seedy, unlbx, unlby, print_loss=True):
        loss = 0
        for i in range(stepnum):
            loss = self.train_per_step(seedx, seedy, unlbx, unlby)
            if i % int(stepnum / 30) == 0 and print_loss:
                print(i, 'th ,loss', loss)
        return loss
    
    def train_per_step(self, sx, sy, ux, uy):
        _, loss = self.sess.run(fetches=[self.trainop, self.loss],
                                feed_dict={self.seedxe: sx, self.seedye: sy, self.unlbxe: ux,
                                           self.unlbye: uy})
        return loss
    
    def predict(self, idfmtx):
        return self.sess.run(self.seedpred, feed_dict={self.seedxe: idfmtx})
    
    def get_theta(self):
        return self.sess.run(self.params)
    
    def load_params(self, file_name):
        saver = tf.train.Saver(self.params)
        saver.restore(self.sess, file_name)
    
    def save_params(self, file_name):
        saver = tf.train.Saver(self.params)
        saver.save(self.sess, file_name)

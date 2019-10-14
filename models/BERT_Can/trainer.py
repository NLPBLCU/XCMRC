import numpy as np
MIN_NUM = -1e9
import tensorflow as tf

class Trainer(object):
    def __init__(self, config, model, word2id):
        self.config = config
        self.model = model
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.word_dict = {word2id[w]: w for w in word2id.keys()}
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch):
        assert isinstance(sess, tf.Session)
        bs = len(batch['doc'])
        feed_dict = {self.model.doc: batch['doc'],
                     self.model.doc_mask: batch['doc_mask'],

                     self.model.que: batch['que'],
                     self.model.que_mask: batch['que_mask'],

                     self.model.y: np.reshape(batch['y'], [len(batch['y'])]).astype(np.float32),

                     self.model.c: np.array(batch['candidates']).astype(np.float32),
                     self.model.is_train: True}

        #passage, ans_repre = sess.run([self.model.passage, self.model.ans_repre], feed_dict=feed_dict)

        loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

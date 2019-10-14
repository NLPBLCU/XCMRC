import numpy as np
MIN_NUM = -1e9
import tensorflow as tf
import logging

class Trainer(object):
    def __init__(self, config, model, word2id):
        #import pdb
        #pdb.set_trace()
        self.config = config
        self.model = model
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)
        self.word_dict = {word2id[w]: w for w in word2id.keys()}

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch):
        import pdb
        #pdb.set_trace()
        assert isinstance(sess, tf.Session)
        bs = len(batch['doc'])
        feed_dict = {
                     self.model.que: batch['que'],
                     self.model.que_mask: batch['que_mask'],

                     self.model.y: np.reshape(batch['y'], [len(batch['y'])]).astype('int32'),

                     self.model.c: batch['candidates'],
                     self.model.is_train: True}
        import pdb
        #pdb.set_trace()
        #seq2seq_outputs, target_sentence, decode_length = sess.run([self.model.seq2seq_outputs, self.model.target_sentence, self.model.decode_length],feed_dict=feed_dict)
        #print(np.shape(seq2seq_outputs))
        #print(np.shape(target_sentence))
        #print(np.shape(decode_length))
        loss, train_op = sess.run([self.model.loss, self.train_op], feed_dict=feed_dict)
        return loss
        #return loss

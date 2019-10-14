import numpy as np
MIN_NUM = -1e9
import tensorflow as tf

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
        assert isinstance(sess, tf.Session)
        ds = batch
        import pdb
        #pdb.set_trace()
        answer_mask = np.zeros_like(batch['doc']).astype(np.float32) + MIN_NUM
        labels = np.zeros_like(batch['doc']).astype(np.float32)
        for i,items in enumerate(batch['doc']):
            for j, item in enumerate(items):
                if item in batch['candidates'][i]:
                    answer_mask[i][j] = 0
                if item == batch['candidates'][i][int(batch['y'][i][0])]:
                    labels[i, j] = 1
        feed_dict = {}
        feed_dict[self.model.doc] = batch['doc']
        feed_dict[self.model.y] = labels
        feed_dict[self.model.doc_mask] = batch['doc_mask']
        feed_dict[self.model.que] = batch['que']
        feed_dict[self.model.que_mask] = batch['que_mask']
        feed_dict[self.model.answer_mask] = answer_mask
        feed_dict[self.model.is_train] = True

        loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

import nltk
import random
import numpy as np
from collections import Counter
import nltk
import re
import tensorflow as tf
import bidaf_can.ops as ops
import my.tensorflow.nn as nn
from my.tensorflow import get_initializer
from tensorflow.python.layers import core as layers_core

class BaselineModel(object):
    def __init__(self,config,vocab_size, emb_mat=None):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N = config.batch_size
        dm = config.word_emb_size
        self.vocab_size = vocab_size
        self.emb_mat = emb_mat
        self.num_units = 512
        self.layers = 2

        # 输入篇章
        self.doc = tf.placeholder('int32', [N, None], name='doc')
        self.doc_mask = tf.placeholder('bool', [N, None], name='doc_mask')

        # 输入问句
        self.que = tf.placeholder('int32', [N, None], name='que')
        self.que_mask = tf.placeholder('int32', [N, None], name='que_mask')

        # 候选项
        self.c = tf.placeholder('int32', [N, None], name='c')
        self.y = tf.placeholder('int32', [N], name='y')

        self.is_train = tf.placeholder('float32', [])
        self.new_emb_mat = tf.placeholder('float32', [None, dm], name='new_word_emb')

        self.is_train = tf.placeholder('bool', [], name='is_train')

        # Define misc
        self.tensor_dict = {}

        # Loss outputs
        self.loss = None
        self.var_list = None
        #pdb.set_trace()
        self._build_forward()
        self._build_loss()


    def _build_forward(self):
        #import pdb
        #pdb.set_trace()
        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                if self.config.mode == 'train':
                    assert self.emb_mat is not None
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[self.vocab_size-np.shape(self.emb_mat)[0],
                                                          self.config.word_emb_size],
                                                   dtype='float', trainable=True)
                    pretrained_emb_mat = tf.get_variable("pretrained_emb_mat", shape=[np.shape(self.emb_mat)[0],
                                                                                      self.config.word_emb_size],
                                                         dtype='float', initializer=get_initializer(self.emb_mat),
                                                         trainable=self.config.fine_tune)
                else:
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[self.vocab_size-np.shape(self.emb_mat)[0],
                                                          self.config.word_emb_size],
                                                   dtype='float')
                    pretrained_emb_mat = tf.get_variable("pretrained_emb_mat",
                                                         shape=[np.shape(self.emb_mat)[0], self.config.word_emb_size],
                                                         dtype='float')
                self.word_emb_mat = tf.concat(values=[word_emb_mat, pretrained_emb_mat], axis=0)

            with tf.name_scope("word"):
                self.doc_word_emb = tf.nn.embedding_lookup(self.word_emb_mat, self.doc)  # [N, M, d]
                self.que_word_emb = tf.nn.embedding_lookup(self.word_emb_mat, self.que)  # [N, M, d]
                self.ans_word_emb = tf.nn.embedding_lookup(self.word_emb_mat, self.c)  # [N, 10, d]

        with tf.variable_scope("sentence_modeling"):
            self.doc_repre, _ = ops.m_cudnn_lstm(inputs=self.doc_word_emb, num_layers=1,
                                                 hidden_size=self.config.hidden_size,
                                                 weight_noise_std=None, is_training=self.is_train, scope='doc',
                                                 input_drop_prob=0, i_m="linear_input",
                                                 use_gru=self.config.use_gru)  # [N,M,JX,d]

            self.que_repre, _ = ops.m_cudnn_lstm(inputs=self.que_word_emb, num_layers=1, hidden_size=self.config.hidden_size,
                                                 weight_noise_std=None, is_training=self.is_train, scope='que',
                                                 input_drop_prob=0, i_m="linear_input",
                                                 use_gru=self.config.use_gru)  # [N,M,JX,d]

            self.attend_passage = self._bi_attention(self.config, self.is_train, self.doc_repre, self.que_repre,
                                              h_mask=self.doc_mask, scope="bi_att", tensor_dict=self.tensor_dict)

            self.passage, _ = ops.m_cudnn_lstm(inputs=self.attend_passage, num_layers=1, hidden_size=self.config.hidden_size,
                                          weight_noise_std=None, is_training=self.is_train, scope='aggregate',
                                          input_drop_prob=0, i_m="linear_input",
                                          use_gru=self.config.use_gru)

        # Concatenate outputs and states of the forward and backward RNNs
        with tf.variable_scope("summary"):
            logit = tf.layers.dense(self.passage, 1, activation=None)
            rc_logits = tf.nn.softmax(logit, 1)
            passage = tf.reduce_sum(self.passage * rc_logits, 1)  # [N,dim]
        with tf.variable_scope('get_answer'):
            self.prediction = self._answer_layer(passage, self.ans_word_emb)

    def _answer_layer(self, doc, candidates):
        'doc # [N,dim] candidates: [N,num_candidates, dim]'
        candidates = nn.highway_network(candidates, self.config.highway_layers, True, is_train=self.is_train)
        candidates = tf.layers.dense(candidates, doc.get_shape().as_list()[-1])
        doc = tf.reshape(doc, [tf.shape(doc)[0], 1, tf.shape(doc)[1]])
        logit = tf.matmul(doc, candidates, transpose_b=True)  # [N, 1, num_candidates]
        return tf.reshape(logit, [tf.shape(logit)[0], tf.shape(logit)[2]])

    def _build_loss(self):
        config = self.config
        N = self.doc.get_shape().as_list()[0]

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(tf.clip_by_value(probs, epsilon, 1e90)), 1)
            return losses

        def log_loss(scores, margin, gamma=2):
            loss = tf.log1p(tf.clip_by_value(tf.exp(gamma * (margin + scores)), 1e-9, 1e90))
            return loss
        self.labels = tf.one_hot(self.y, tf.shape(self.prediction)[1], axis=1)
        self.mono_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.prediction))

        tf.add_to_collection('losses', self.mono_loss)

        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
        if config.add_l2_loss:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss += config.l2_regularizer * l2_loss

    def _bi_attention(self, config, is_train, h, u, h_mask=None, scope=None, tensor_dict=None):
        dim = u.get_shape().as_list()[-1]
        transformed_h = tf.layers.dense(h, dim)
        with tf.variable_scope(scope or "bi_attention"):
            sim_matrix = tf.matmul(transformed_h, u, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), u)
            concat_outputs = tf.concat([h, context2question_attn, transformed_h * context2question_attn], -1)
            concat_outputs *= tf.cast(tf.expand_dims(h_mask, -1), 'float32')
            return concat_outputs

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

import nltk
import random
import numpy as np
from collections import Counter
import nltk
import re
import tensorflow as tf
import BiDAF_Cloze.ops as ops
import my.tensorflow.nn as nn
from my.tensorflow import get_initializer

class BaselineModel(object):
    def __init__(self,config,vocab_size, emb_mat=None):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N = config.batch_size
        dm = config.word_emb_size
        self.emb_mat = emb_mat
        self.vocab_size = vocab_size
        # 输入的是一整个篇章
        self.doc = tf.placeholder('int32', [N, None], name='passage')
        self.doc_mask = tf.placeholder('bool', [N, None], name='x_mask')
        self.answer_mask = tf.placeholder('float32', [N, None], name='answer_mask')

        # 十八个类别的解释
        self.que = tf.placeholder('int32', [N, None], name='q')
        self.que_mask = tf.placeholder('bool', [N, None], name='q_mask')

        self.y = tf.placeholder('float32', [N, None], name='y')

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
                                                   shape=[self.vocab_size-np.shape(self.emb_mat)[0], self.config.word_emb_size],
                                                   dtype='float', trainable=True)
                    pretrained_emb_mat = tf.get_variable("pretrained_emb_mat",
                                                         shape=[np.shape(self.emb_mat)[0], self.config.word_emb_size],
                                                         dtype='float', initializer=get_initializer(self.emb_mat),
                                                         trainable=self.config.fine_tune)
                else:
                    word_emb_mat = tf.get_variable("word_emb_mat", shape=[self.vocab_size-np.shape(self.emb_mat)[0], self.config.word_emb_size],dtype='float')
                    pretrained_emb_mat = tf.get_variable("pretrained_emb_mat",
                                                         shape=[np.shape(self.emb_mat)[0], self.config.word_emb_size],
                                                         dtype='float')
                word_emb_mat = tf.concat(values=[word_emb_mat, pretrained_emb_mat],axis=0)
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.doc)  # [N, M, JX, d]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.que)  # [N, JQ, d]

        passage, _ = ops.m_cudnn_lstm(inputs=Ax, num_layers=1, hidden_size=self.config.hidden_size,
                                      weight_noise_std=None, is_training=self.is_train, scope='doc',
                                      input_drop_prob=0, i_m="linear_input",
                                      use_gru=self.config.use_gru)  # [N,M,JX,d]
        question, _ = ops.m_cudnn_lstm(inputs=Aq, num_layers=1, hidden_size=self.config.hidden_size,
                                  weight_noise_std=None, is_training=self.is_train, scope='question',
                                  input_drop_prob=0, i_m="linear_input",
                                  use_gru=self.config.use_gru)  # [N,M,JX,d]
        with tf.variable_scope("bidaf"):
            passage = self.bi_attention(self.config, self.is_train, passage, question, self.doc_mask)
        passage, _ = ops.m_cudnn_lstm(inputs=passage, num_layers=1, hidden_size=self.config.hidden_size,
                                      weight_noise_std=None, is_training=self.is_train, scope='aggregate',
                                      input_drop_prob=0, i_m="linear_input",
                                      use_gru=self.config.use_gru)
        with tf.variable_scope('get_answer'):
            prediction = tf.layers.dense(passage, 1)
            prediction = tf.reshape(prediction, [tf.shape(prediction)[0], tf.shape(prediction)[1]])
            self.prediction = prediction + self.answer_mask

    def _answer_layer(self, doc, candidates):
        'doc # [N,dim] candidates: [N,num_candidates, dim]'
        candidates = nn.highway_network(candidates,2, True,is_train=self.is_train)
        candidates = tf.layers.dense(candidates, doc.get_shape().as_list()[-1])
        doc = tf.reshape(doc,[tf.shape(doc)[0], 1, tf.shape(doc)[1]])
        logit = tf.matmul(doc, candidates, transpose_b=True) # [N, 1, num_candidates]
        return tf.reshape(logit, [tf.shape(logit)[0], tf.shape(logit)[2]])

    def _build_loss(self):
        config = self.config
        N = self.doc.get_shape().as_list()[0]

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        tf.add_to_collection('losses', loss)

        self.loss = tf.reduce_mean(tf.add_n(tf.get_collection('losses'), name='loss'))
        if config.add_l2_loss:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss += config.l2_regularizer * l2_loss

    def bi_attention(self, config, is_train, h, u, h_mask=None, scope=None, tensor_dict=None):
        dim = u.get_shape().as_list()[-1]
        transformed_h = tf.layers.dense(h, dim)
        with tf.variable_scope(scope or "bi_attention"):
            sim_matrix = tf.matmul(transformed_h, u, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), u)
            concat_outputs = tf.concat([h, context2question_attn,
                                            transformed_h * context2question_attn], -1)
            concat_outputs *= tf.cast(tf.expand_dims(h_mask, -1), 'float32')
            return concat_outputs

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

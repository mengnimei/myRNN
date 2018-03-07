#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        
        self.global_step = tf.Variable(0, trainable=False, name='self.global_step', dtype=tf.int64)# 存放运行到第几个step（global step）
        self.X = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='input')#存放input数据X
        self.Y = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='label')#存放groundtruth（label）即Y
        self.keep_prob = tf.placeholder(1, name='self.keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])   #[5000,128]
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)
        


        
        with tf.variable_scope('rnn'):
            ##################
            # My Code here
            ##################
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_embedding, forget_bias=0.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.rnn_layers, state_is_tuple=True)
            self.state_tensor = cell.zero_state(self.batch_size, tf.float32)
            outputs_tensor, self.outputs_state_tensor= tf.nn.dynamic_rnn(cell, data, initial_state=self.state_tensor)
            print(outputs_tensor)
            print(self.outputs_state_tensor)

            ##################
            # My Code End
            ##################
        # concate every time step
        seq_output = tf.concat(outputs_tensor, 1)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            ##################
            # My Code here
            ##################

            W = tf.get_variable('W', [self.dim_embedding, self.num_words], dtype=tf.float32)
            b = tf.get_variable('b', [self.num_words], dtype=tf.float32)
            logits = tf.matmul(seq_output_final, W) + b
            ##################
            # My Code End
            ##################

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        y_one_hot = tf.one_hot(self.Y, self.num_words)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()

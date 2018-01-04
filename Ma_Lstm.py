import tensorflow as tf
from tensorflow.contrib import rnn
from helper import last_relevant_output
import numpy as np

class model(object):

    def __init__(
            self,
            max_sequence_length,
            total_classes,
            embedding_size,
            id2Vecs,
            batch_size,
            lmd = 0
                 ):

            self.sent1 = tf.placeholder(tf.int32, [batch_size, max_sequence_length], name="sent1")
            self.sent1_length = tf.placeholder(tf.int32,[batch_size] , name = "sent1_length")
            self.sent2 = tf.placeholder(tf.int32, [batch_size, max_sequence_length], name="sent2")
            self.sent2_length = tf.placeholder(tf.int32, [batch_size], name="sent2_length")

            ## labels
            self.labels = tf.placeholder(tf.float32, [batch_size], name="labels")
            ## dropout
            self.dropout = tf.placeholder(tf.float32, name="dropout")
            ## hidden units
            self.hidden_Units = 50
            self.batch_size = batch_size
            self.id2Vecs = id2Vecs
            self.embedding_size = embedding_size
            self.id2Vecs = id2Vecs

            self.l2_loss = tf.constant(value=0.0, dtype=tf.float32)
            with tf.variable_scope('this-scope') as scope:
                self.right_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
                self.left_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)

                self.right_lstm_cell = rnn.DropoutWrapper(self.right_lstm_cell, output_keep_prob=self.dropout)
                self.left_lstm_cell = rnn.DropoutWrapper(self.left_lstm_cell, output_keep_prob=self.dropout)
                sent_1 = self.get_word_emb(self.sent1, name="sent_1")
                scope.reuse_variables()
                sent_2 = self.get_word_emb(self.sent2, name="sent_2")

            (fw_out_1,bw_out_1), _  = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.left_lstm_cell,cell_bw= self.right_lstm_cell,
                                                                       inputs=sent_1,
                                                                       sequence_length=self.sent1_length,
                                                                       dtype=tf.float32)
            (fw_out_2,bw_out_2), _  = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.left_lstm_cell,
                                                                           cell_bw=self.right_lstm_cell,
                                                                           inputs=sent_2,
                                                                           sequence_length= self.sent2_length,
                                                                           dtype=tf.float32)

            last_fw_out_1 = last_relevant_output(fw_out_1,self.sent1_length)
            last_bw_out_1 = last_relevant_output(bw_out_1, self.sent1_length)

            last_fw_out_2 = last_relevant_output(fw_out_2, self.sent2_length)
            last_bw_out_2 = last_relevant_output(bw_out_2, self.sent2_length)

            # sent_1_repr = tf.layers.dropout(tf.concat([last_fw_out_1,last_bw_out_1], 1),rate=self.dropout)
            # sent_2_repr = tf.layers.dropout(tf.concat([last_fw_out_2,last_bw_out_2],1),rate=self.dropout)
            sent_1_repr = tf.concat([last_fw_out_1, last_bw_out_1], 1)
            sent_2_repr = tf.concat([last_fw_out_2, last_bw_out_2], 1)
            abs_diff = tf.abs(tf.subtract(sent_1_repr,sent_2_repr))
            sum_ = tf.reduce_sum(abs_diff,axis=1)

            # scale
            self.final_score = 1.0 + 4*tf.exp(-sum_)
            # self.final_score = sum_
            # self.labels = (self.labels - 1.0)/4.0
            with tf.name_scope('loss'):
                self.loss = tf.losses.mean_squared_error(self.final_score, self.labels)
            with tf.name_scope('accuracy'):
                self.acc = self.pearson_correlation(self.final_score, self.labels)


    def pearson_correlation(self, x, y):
        """
        need to test this function
        :param x: tensor 1
        :param y: tensor 2
        :return: pearson coefficient
        """
        numerator = (self.batch_size * tf.reduce_sum(x * y) - tf.reduce_sum(x) * tf.reduce_sum(y))
        denominator = tf.sqrt((self.batch_size * tf.reduce_sum(x * x) - tf.reduce_sum(x) * tf.reduce_sum(x)) *
                              (self.batch_size * tf.reduce_sum(y * y) - tf.reduce_sum(y) * tf.reduce_sum(y)))
        return numerator / denominator

    def get_word_emb(self, x, name):
        """
        :param x:
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope("word-embedding-layer"):
                self.embeddings = tf.Variable(initial_value=self.id2Vecs, dtype=tf.float32, name='embedding_lookup',trainable=False)
                word_embeddings = tf.nn.embedding_lookup(self.embeddings, x, name=name)

        return word_embeddings

# m = model(max_sequence_length=13,total_classes=1,embedding_size=300,id2Vecs=np.zeros([10,300]),batch_size=60)
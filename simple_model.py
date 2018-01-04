import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from swish_activation import swish
from tensorflow.contrib.metrics import streaming_pearson_correlation


class model(object):
    def __init__(
            self, max_sequence_length,
            total_classes,
            embedding_size,
            char_size,
            char_embed_size,
            id2Vecs,
            max_word_len,
            sent_dim,
            threshold=0.5,
            lmd=1e-4
    ):
        # placeholders
        ## sentence 1
        self.x1 = tf.placeholder(tf.int32, [None, max_sequence_length], name="x")
        self.chars_x1 = tf.placeholder(tf.int32, [None, max_sequence_length, max_word_len])
        self.sent1 = tf.placeholder(tf.float32,[None,sent_dim])
        ## sentence 2
        self.x2 = tf.placeholder(tf.int32, [None, max_sequence_length], name="x")
        self.chars_x2 = tf.placeholder(tf.int32, [None, max_sequence_length, max_word_len])
        self.sent2 = tf.placeholder(tf.float32, [None, sent_dim])
        ## labels
        self.labels = tf.placeholder(tf.float32, [None], name="labels")
        ## dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
        ## hidden units
        # self.hidden_Units = 50
        self.char_embed_size = char_embed_size
        # self.batch_size = batch_size
        self.id2Vecs = id2Vecs
        # tf.split() might be useful
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.char_size = char_size
        self.max_word_len = max_word_len
        self.total_classes = total_classes
        self.char_embed_size = char_embed_size
        self.lmd = lmd
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 50
        self.total_filters = self.num_filters * len(self.filter_sizes)
        self.threshold = threshold
        self.hidden_Units = 100
        self.l2_loss = tf.constant(value=0.0, dtype=tf.float32)
        ## change activation function here
        # self.activation  = swish
        with tf.device('/cpu:0'):
            #     for i, filter_size in enumerate(self.filter_sizes):
            #         with tf.name_scope("conv-maxpool-%s" % filter_size) as scope:
            #             # Convolution Layer
            #             filter_shape = [filter_size, self.char_embed_size, 1, self.num_filters]
            #             self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            #             self.b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")


            #
            with tf.variable_scope('this-scope') as scope:
                # self.right_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
                # self.left_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
                #
                # self.right_lstm_cell = rnn.DropoutWrapper(self.right_lstm_cell, output_keep_prob=1.0 - self.dropout)
                # self.left_lstm_cell = rnn.DropoutWrapper(self.left_lstm_cell, output_keep_prob=1.0 - self.dropout)
                # sent_1 = self.get_text_emb(self.x1, name="sent_1")
                # scope.reuse_variables()
                # sent_2 = self.get_text_emb(self.x1,name="sent_2")
                # scope._reuse = None
                # l = []
                # out1 = self.get_out(sent_1)
                # out2 = self.get_out(sent_2)
                out1 = self.sent1
                out2 = self.sent2
                # l.append(out1)
                # l.append(out2)
                dot_ = tf.layers.dropout(tf.multiply(out1,out2),rate= 1.0 - self.dropout_keep_prob)
                diff_ = tf.layers.dropout(tf.abs(tf.subtract(out1,out2)),rate= 1.0 - self.dropout_keep_prob)
                exp_diff = tf.layers.dropout(tf.exp(-tf.abs(tf.subtract(out1,out2))),rate= 1.0 - self.dropout_keep_prob)
                out1_ = tf.layers.dropout(out1,rate= 1.0 - self.dropout_keep_prob)
                out2_ = tf.layers.dropout(out2,rate= 1.0 - self.dropout_keep_prob)

                with tf.name_scope("last-layer"):
                    self.W_f_1 = tf.get_variable(
                        "W_f_1",
                        shape=[sent_dim, self.total_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                    self.W_f_2 = tf.get_variable(
                        "W_f_2",
                        shape=[sent_dim, self.total_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                    self.W_f_3 = tf.get_variable(
                        "W_f_3",
                        shape=[sent_dim, self.total_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                    self.W_f_4 = tf.get_variable(
                        "W_f_4",
                        shape=[sent_dim, self.total_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                    self.W_f_5 = tf.get_variable(
                        "W_f_5",
                        shape=[sent_dim, self.total_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )


                    bias = tf.Variable(tf.constant(value=0.01, shape=[self.total_classes], name="bias"))
                    self.l2_loss += tf.nn.l2_loss(self.W_f_1) + tf.nn.l2_loss(self.W_f_2) + tf.nn.l2_loss(self.W_f_3) \
                                    + tf.nn.l2_loss(self.W_f_4) + tf.nn.l2_loss(self.W_f_5)
                    self.l2_loss += tf.nn.l2_loss(bias)
                    final_score = tf.matmul(dot_,self.W_f_1) + tf.matmul(diff_,self.W_f_2) + bias \
                                       + tf.matmul(out1_,self.W_f_3) + tf.matmul(out2_,self.W_f_4)  \
                                        # + tf.matmul(exp_diff,self.W_f_5)
                    self.final_score = final_score
                    # self.final_score = tf.matmul( tf.reduce_sum(exp_diff,axis=1),self.W_f_6)
                    self.final_score = 4*tf.sigmoid(tf.reshape(self.final_score,[-1])) + 1
                with tf.name_scope("loss"):
                    self.loss = tf.losses.mean_squared_error(self.final_score, self.labels)
                    self.loss += lmd * self.l2_loss
            ## onjective function --> pearson correlation
            with tf.name_scope("accuracy"):
                self.acc = self.pearson_correlation(self.final_score, self.labels)

    def pearson_correlation(self, x, y):
        """
        need to test this function
        :param x: tensor 1
        :param y: tensor 2
        :return: pearson coefficient
        """
        self.batch_size = tf.cast(tf.size(y),dtype=tf.float32)
        numerator = (self.batch_size * tf.reduce_sum(x * y) - tf.reduce_sum(x) * tf.reduce_sum(y))
        denominator = tf.sqrt((self.batch_size * tf.reduce_sum(x * x) - tf.reduce_sum(x) * tf.reduce_sum(x)) *
                              (self.batch_size * tf.reduce_sum(y * y) - tf.reduce_sum(y) * tf.reduce_sum(y)))
        return numerator / denominator

    def get_text_emb(self, x,name):
        word_emb_1 = self.get_word_emb(x, name + "_word_1")

        return word_emb_1


    def get_word_emb(self, x, name):
        """
        :param x:
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope("word-embedding-layer"):
                self.embeddings = tf.Variable(initial_value=self.id2Vecs, dtype=tf.float32, name='embedding_lookup')
                word_embeddings = tf.nn.embedding_lookup(self.embeddings, x, name=name)

        return word_embeddings

    def get_out(self, sentences):
        self.outputs, self.state = tf.nn.bidirectional_dynamic_rnn(self.left_lstm_cell, self.right_lstm_cell, sentences,
                                                                   dtype=tf.float32)
        combined_output = tf.concat(self.outputs, axis=2)
        out = tf.reshape(combined_output, shape=[self.batch_size, self.max_sequence_length, self.hidden_Units * 2])
        # out = combined_output[:,-1,:]
        return out


        # convert data to slices

        # Initial state of the LSTM memory.
        # hidden_state = tf.zeros([self.batch_size,])
        #
        # self.sequence = tf.split(self.sentences,num_or_size_splits=max_sequence_length,axis=1)


# m = model(max_sequence_length=11,total_classes=1,embedding_size=300,char_size = 357,max_word_len =12 ,char_embed_size = 30,id2Vecs=np.zeros([7,300]),batch_size=3)
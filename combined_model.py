import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from swish_activation import swish
from tensorflow.contrib.metrics import streaming_pearson_correlation
from helper import last_relevant_output

class model(object):
    def __init__(
            self, max_sequence_length,
            total_classes,
            embedding_size,
            id2Vecs,
            batch_size,
            sentence_dimension,
            threshold=0.5,
            lmd=1e-4
    ):
        # placeholders
        self.sent1 = tf.placeholder(tf.int32, [None, max_sequence_length], name="sent1")
        self.sent1_length = tf.placeholder(tf.int32, [None], name="sent1_length")
        self.sent1_enc = tf.placeholder(tf.float32, [None, sentence_dimension],name='sent1_enc')

        self.sent2 = tf.placeholder(tf.int32, [None, max_sequence_length], name="sent2")
        self.sent2_length = tf.placeholder(tf.int32, [None], name="sent2_length")
        self.sent2_enc = tf.placeholder(tf.float32, [None, sentence_dimension],name='sent2_enc')

        self.embedding_size = embedding_size
        self.max_sequence_length  =max_sequence_length
        ## labels
        self.labels = tf.placeholder(tf.float32, [None], name="labels")
        ## dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
        ## hidden units
        self.hidden_Units = 50
        self.total_classes = total_classes
        # self.batch_size = batch_size
        self.id2Vecs = id2Vecs
        self.embedding_size = embedding_size
        self.id2Vecs = id2Vecs
        self.sent_dim = 2*self.hidden_Units
        self.l2_loss = tf.constant(value=0.0, dtype=tf.float32)
        with tf.variable_scope('this-scope') as scope:
            self.right_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
            self.left_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)

            self.right_lstm_cell = rnn.DropoutWrapper(self.right_lstm_cell, output_keep_prob=self.dropout_keep_prob)
            self.left_lstm_cell = rnn.DropoutWrapper(self.left_lstm_cell, output_keep_prob=self.dropout_keep_prob)
            sent_1 = self.get_word_emb(self.sent1, name="sent_1")
            scope.reuse_variables()
            sent_2 = self.get_word_emb(self.sent2, name="sent_2")
        (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.left_lstm_cell,
                                                                  cell_bw=self.right_lstm_cell,
                                                                  inputs=sent_1,
                                                                  sequence_length=self.sent1_length,
                                                                  dtype=tf.float32)
        (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.left_lstm_cell,
                                                                  cell_bw=self.right_lstm_cell,
                                                                  inputs=sent_2,
                                                                  sequence_length=self.sent2_length,
                                                                  dtype=tf.float32)

        # last_fw_out_1 = last_relevant_output(fw_out_1, self.sent1_length)
        # last_bw_out_1 = last_relevant_output(bw_out_1, self.sent1_length)
        #
        # last_fw_out_2 = last_relevant_output(fw_out_2, self.sent2_length)
        # last_bw_out_2 = last_relevant_output(bw_out_2, self.sent2_length)

        # sent_1_repr = tf.layers.dropout(tf.concat([last_fw_out_1,last_bw_out_1], 1),rate=self.dropout)
        # sent_2_repr = tf.layers.dropout(tf.concat([last_fw_out_2,last_bw_out_2],1),rate=self.dropout)
        # sent_1_repr = tf.concat([last_fw_out_1, last_bw_out_1], 1)
        # sent_2_repr = tf.concat([last_fw_out_2, last_bw_out_2], 1)
        combined_output_1 = tf.concat([fw_out_1,bw_out_1], axis=2)
        out1 = tf.reshape(combined_output_1, shape=[-1, max_sequence_length * self.hidden_Units * 2])
        combined_output_2 = tf.concat([fw_out_2, bw_out_2], axis=2)
        out2 = tf.reshape(combined_output_2, shape=[-1, max_sequence_length * self.hidden_Units * 2])

        # self.final_score = self.similarity_score(self.sent1_enc,self.sent2_enc)

        # exp_sum_diff = tf.layers.dropout(self.sent1_enc*self.sent2_enc , rate= 1.0-self.dropout_keep_prob)
            # tf.exp(tf.abs(self.sent1_enc - self.sent2_enc))

        self.sent_dim = 2 * self.hidden_Units * self.max_sequence_length
        out1 = combined_output_1
        out2 = combined_output_2
        out1 = tf.concat([tf.reshape(out1, [-1, self.sent_dim]), self.sent1_enc], axis=1)
        out2 = tf.concat([tf.reshape(out2, [-1, self.sent_dim]), self.sent2_enc], axis=1)
        self.sent_dim += sentence_dimension
        # attn = self.bi_attention(out1,out2)
        # attn = tf.reshape(attn,[-1,self.sent_dim])
        # out1 = self.sent1_enc
        # out2 = self.sent2_enc
        # self.sent_dim = 4096

        # l.append(out1)
        # l.append(out2)
        self.dot_ = tf.layers.dropout(tf.multiply(out1,out2),rate=1.0 - self.dropout_keep_prob)
        self.diff_ = tf.layers.dropout(tf.abs(tf.subtract(out1,out2)),rate=1.0 - self.dropout_keep_prob)
        self.exp_diff = tf.layers.dropout(tf.exp(-tf.abs(tf.subtract(out1,out2))),rate= 1.0 - self.dropout_keep_prob)
        self.out1_ = tf.layers.dropout(out1,rate=1.0 - self.dropout_keep_prob)
        self.out2_ = tf.layers.dropout(out2,rate=1.0 - self.dropout_keep_prob)

        with tf.name_scope("last-layer"):
            self.W_f_1 = tf.get_variable(
                "W_f_1",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W_f_2 = tf.get_variable(
                "W_f_2",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W_f_3 = tf.get_variable(
                "W_f_3",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W_f_4 = tf.get_variable(
                "W_f_4",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W_f_5 = tf.get_variable(
                "W_f_5",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            self.W_f_6 = tf.get_variable(
                "W_f_6",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            # self.W_f_7 = tf.get_variable(
            #     "W_f_7",
            #     shape=[4096,total_classes],
                # initializer=tf.contrib.layers.xavier_initializer()
        #     )
            bias = tf.Variable(tf.constant(value=0.01, shape=[self.total_classes], name="bias"))
            self.l2_loss += tf.nn.l2_loss(self.W_f_1) + tf.nn.l2_loss(self.W_f_2) + tf.nn.l2_loss(self.W_f_3) \
                            + tf.nn.l2_loss(self.W_f_4) + tf.nn.l2_loss(self.W_f_5) + tf.nn.l2_loss(self.W_f_6)
            self.l2_loss += tf.nn.l2_loss(bias)
            final_score = tf.matmul(self.dot_,self.W_f_1) + tf.matmul(self.diff_,self.W_f_2) + bias \
                               + tf.matmul(self.out1_,self.W_f_3) + tf.matmul(self.out2_,self.W_f_4)  \
                                    # + tf.matmul(self.exp_diff,self.W_f_5) \
                          # + tf.matmul(self.cosine_sim,self.W_f_6)\
                                    # + tf.matmul(exp_sum_diff,self.W_f_7)
            # final_score = tf.matmul(attn,self.W_f_6) + bias
            self.final_score = 1.0 + 4*tf.sigmoid(final_score)
            self.final_score = tf.reshape(self.final_score,[-1])
        with tf.name_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.final_score, self.labels)
            self.loss += lmd * self.l2_loss
        tf.summary.scalar('loss', self.loss)
        ## onjective function --> pearson correlation
        with tf.name_scope("accuracy"):
            self.acc = self.pearson_correlation(self.final_score, self.labels)
        tf.summary.scalar('accuracy', self.acc)






    def variable_summaries(self,var_):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var_)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var_ - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var_))
            tf.summary.scalar('min', tf.reduce_min(var_))
            tf.summary.histogram('histogram', var_)

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

    def bi_attention(self,x,y):
        # matrx = tf.matmul(x,y,transpose_b=True)
        print("attention")
        x = tf.reshape(x,[self.batch_size,self.max_sequence_length,2*self.hidden_Units])
        y = tf.reshape(y, [self.batch_size, self.max_sequence_length, 2 * self.hidden_Units])

        W1 = tf.get_variable("W1",shape=[2*self.hidden_Units,1],initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2",shape=[2*self.hidden_Units,1],initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable("W3",shape=[2*self.hidden_Units,1],initializer=tf.contrib.layers.xavier_initializer())

        out_ = []
        for l in range(self.batch_size):
            # S = tf.zeros(shape=[self.max_sequence_length, self.max_sequence_length])
            S = []
            for i in range(self.max_sequence_length):
                s = []
                for j in range(self.max_sequence_length):
                    s.append( tf.matmul(tf.reshape(x[l,i,:],[1,2*self.hidden_Units]),W1)
                              + tf.matmul(tf.reshape(y[l,j,:],[1,2*self.hidden_Units]),W2)
                              +  tf.matmul(tf.reshape(tf.multiply(x[l,i,:],y[l,j,:]), [1, 2 * self.hidden_Units]), W3)
                              )
                S.append(s)
            S = tf.squeeze(tf.convert_to_tensor(S))
            a = tf.nn.softmax(S,dim=1)
            o1 = []
            for i in range(self.max_sequence_length):
                o2 = []
                for j in range(self.max_sequence_length):
                    o2.append(a[i,j]*x[l,j,:])
                o1.append(o2)
            print("batch : "+ str(l))
            out_.append(o1)

        out_ = tf.reduce_sum(tf.convert_to_tensor(out_),axis=2)
        return out_
    def similarity_score(self,x,y):
        concat_x_y = tf.concat([x,y],axis=1)
        self.W_1 = tf.get_variable(
                "W_1",
                shape=[4096*2,1024],
                initializer=tf.contrib.layers.xavier_initializer()
            )
        out1 = tf.nn.sigmoid(tf.matmul(concat_x_y,self.W_1))
        out1 = tf.layers.dropout(out1,rate=1.0-self.dropout_keep_prob )
        self.W_2 = tf.get_variable(
            "W_2",
            shape=[1024,128],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        out2 = tf.nn.sigmoid(tf.matmul(out1,self.W_2))
        out2 = tf.layers.dropout(out2,1.0 - self.dropout_keep_prob )
        self.W_3 = tf.get_variable(
            "W_3",
            shape=[128,1],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        out3 = tf.matmul(out2,self.W_3)
        out3 = tf.reshape(out3,[-1])
        return out3

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

    def get_out(self, sentences,sent_length):
        (fw_out,bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.left_lstm_cell, self.right_lstm_cell, sentences,sequence_length=sent_length,
                                                                   dtype=tf.float32)
        fw_out_last = last_relevant_output(fw_out, sent_length)
        bw_out_last = last_relevant_output(bw_out,sent_length)
        combined_output = tf.concat([fw_out_last,bw_out_last], axis=1)
        # out = tf.reshape(combined_output, shape=[self.batch_size, self.max_sequence_length, self.hidden_Units * 2])
        # out = combined_output[:,-1,:]
        return combined_output


        # convert data to slices

        # Initial state of the LSTM memory.
        # hidden_state = tf.zeros([self.batch_size,])
        #
        # self.sequence = tf.split(self.sentences,num_or_size_splits=max_sequence_length,axis=1)


# m = model(max_sequence_length=11,total_classes=1,embedding_size=300,id2Vecs=np.zeros([7,300]),sentence_dimension=8200,batch_size=3)
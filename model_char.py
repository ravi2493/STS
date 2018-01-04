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
                    batch_size,
                    max_word_len,
                    threshold=0.5,
                    lmd = 0
                 ):
        # placeholders
        ## sentence 1
        self.x1 = tf.placeholder(tf.int32,[batch_size,max_sequence_length],name = "x")
        self.chars_x1 = tf.placeholder(tf.int32,[batch_size,max_sequence_length,max_word_len])
        ## sentence 2
        self.x2 = tf.placeholder(tf.int32, [batch_size, max_sequence_length], name="x")
        self.chars_x2 = tf.placeholder(tf.int32, [batch_size, max_sequence_length, max_word_len])
        ## labels
        self.labels = tf.placeholder(tf.float32,[batch_size,total_classes],name="labels")
        ## dropout
        self.dropout = tf.placeholder(tf.float32,name="dropout")
        ## hidden units
        self.hidden_Units = 100
        self.char_embed_size = char_embed_size
        self.batch_size = batch_size
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
                self.right_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
                self.left_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)

                self.right_lstm_cell = rnn.DropoutWrapper(self.right_lstm_cell, output_keep_prob=self.dropout)
                self.left_lstm_cell = rnn.DropoutWrapper(self.left_lstm_cell, output_keep_prob=self.dropout)
                sent_1 = self.get_text_emb(self.x1,self.chars_x1,name="sent_1")
                scope.reuse_variables()
                sent_2 = self.get_text_emb(self.x1, self.chars_x1,name="sent_2")
                scope._reuse = None
                out1 = self.get_out(sent_1)
                out2 = self.get_out(sent_2)

                # normalize_out1 = tf.nn.l2_normalize(out1, 1)
                # normalize_out2 = tf.nn.l2_normalize(out2, 1)
                out_matrix = []
                for i in range(self.batch_size):
                    out_matrix.append(tf.matmul(out1[i,:,:],tf.transpose(out2[i,:,:])))
                # out = tf.multiply(out1,out2)
                out_matrix = tf.reshape(tf.stack(out_matrix),[-1,self.max_sequence_length*self.max_sequence_length])
                with tf.name_scope("last-layer"):
                    self.W_f = tf.get_variable(
                        "W_f",
                        shape=[self.max_sequence_length * self.max_sequence_length, self.total_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                    )
                    bias = tf.Variable(tf.constant(value=0.01,shape=[self.total_classes],name="bias"))
                    self.l2_loss += tf.nn.l2_loss(self.W_f)
                    self.l2_loss += tf.nn.l2_loss(bias)
                    self.final_score = tf.nn.xw_plus_b(out_matrix,weights=self.W_f,biases=bias,name="scores")

                with tf.name_scope("loss"):
                    self.loss = tf.losses.mean_squared_error(self.final_score, self.labels)
                    self.loss += lmd * self.l2_loss
            ## onjective function --> pearson correlation
            with tf.name_scope("accuracy"):
                # self.acc = tf.contrib.metrics.streaming_pearson_correlation(self.final_score, self.labels)[0]
                self.acc = self.pearson_correlation(self.final_score,self.labels)
                # self.acc = tf.reduce_mean(l)
    def pearson_correlation(self,x,y):
        """
        need to test this function
        :param x: tensor 1
        :param y: tensor 2
        :return: pearson coefficient
        """
        numerator = ( self.batch_size*tf.reduce_sum(x*y) - tf.reduce_sum(x)*tf.reduce_sum(y) )
        denominator = tf.sqrt( (self.batch_size*tf.reduce_sum(x*x) - tf.reduce_sum(x)*tf.reduce_sum(x) )*
                               (self.batch_size*tf.reduce_sum(y*y) - tf.reduce_sum(y)*tf.reduce_sum(y) ) )
        return numerator/denominator
    def get_text_emb(self,x,chars_x,name):

        sent_chars_1 = self.get_char_emb(chars_x)
        word_emb_1 = self.get_word_emb(x,name+"_char_1")

        return self.combine_char_word_emb(sent_chars_1,word_emb_1)

    def get_char_emb(self,chars_x):


        # layer contains trainable weights
        with tf.name_scope("char-embeddings-layer"):

            self.char_embeddings = tf.get_variable(
                                                    "char_embeddings",
                                                    shape=[self.char_size,self.char_embed_size],
                                                    initializer=tf.contrib.layers.xavier_initializer()
                                                )
            self.embedded_chars = tf.nn.embedding_lookup(self.char_embeddings, chars_x)
            self.embedded_chars = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, self.max_word_len, self.char_embed_size, 1])

            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.char_embed_size, 1, self.num_filters]
                    W = tf.get_variable(name="W_"+str(filter_size),shape=filter_shape)
                    b = tf.get_variable(name="b_"+str(filter_size),shape=[self.num_filters])
                    conv = tf.nn.conv2d(
                        self.embedded_chars,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_word_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

        char2word_embeddings = tf.reshape(tf.squeeze(tf.concat(pooled_outputs, axis=3)),[self.batch_size,self.max_sequence_length,self.total_filters])

        return char2word_embeddings

    def get_word_emb(self,x,name):
        """
        :param x:
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope("word-embedding-layer"):
                self.embeddings = tf.Variable(initial_value=self.id2Vecs,dtype=tf.float32,name='embedding_lookup')
                word_embeddings = tf.nn.embedding_lookup(self.embeddings,x,name=name)

        return word_embeddings

    def get_out(self,sentences):
        self.outputs,self.state = tf.nn.bidirectional_dynamic_rnn(self.left_lstm_cell,self.right_lstm_cell,sentences,dtype=tf.float32)
        combined_output = tf.concat(self.outputs, axis=2)
        out = tf.reshape(combined_output,shape=[self.batch_size,self.max_sequence_length,self.hidden_Units*2])
        # out = combined_output[:,-1,:]
        return out

    def combine_char_word_emb(self,chars_emb,word_emb):
        """
        
        :param chars_emb: 
        :param word_emb: 
        :return: combine char and word embeddings
        """""

        sentences =tf.concat(
                [
                    tf.reshape(word_emb,[-1,self.batch_size,self.max_sequence_length,self.embedding_size])
                    ,
                    tf.reshape(chars_emb,[-1,self.batch_size,self.max_sequence_length,self.total_filters])
                ],
                axis=3
            )

        sentences = tf.squeeze(sentences)
        sentences = tf.layers.dropout(sentences,rate=self.dropout)

        return sentences


        # convert data to slices

        #Initial state of the LSTM memory.
        # hidden_state = tf.zeros([self.batch_size,])
        #
        # self.sequence = tf.split(self.sentences,num_or_size_splits=max_sequence_length,axis=1)


# m = model(max_sequence_length=11,total_classes=5,embedding_size=300,char_size = 357,max_word_len =12 ,char_embed_size = 30,id2Vecs=np.zeros([7,300]),batch_size=3)
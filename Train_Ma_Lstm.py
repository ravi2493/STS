from Ma_Lstm import model
import numpy as np
import tensorflow as tf
from preprocess import load_data,load_test_data
from glove import word_embedings
import random
import pickle

def batch_iter(data, batch_size, epochs, Isshuffle=True):
    ## check inputs
    assert isinstance(batch_size,int)
    assert isinstance(epochs,int)
    assert isinstance(Isshuffle,bool)

    num_batches = int((len(data)-1)/batch_size)
    ## data padded
    # data = np.array(data+data[:2*batch_size])
    data_size = len(data)
    # print("size of data"+str(data_size)+"---"+str(len(data)))
    for ep in range(epochs):
        if Isshuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield shuffled_data[start_index:end_index]



def train(m,train_sent_1,train_sent_2,train_len1,train_len2,label,epochs=2000,learning_rate=0.001,check_point=25):
    # if model_name == 'biLstm':
    #     assert isinstance(m,model_bi)
    # else:
    #     assert isinstance(m,model)
    assert isinstance(epochs,int)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(m.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    sess = tf.Session(config=session_conf)
    saver = tf.train.Saver()
    ## intialize
    # sess.run(tf.global_variables_initializer())
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    # dt = np.zeros([len(data),len(data[0])],dtype=int)
    # dt_c = np.zeros([len(data_chars),len(data_chars[0]),len(data_chars[0])])
    # for i,d in enumerate(data):
    #     dt[i,:] = d
    # for i,d in enumerate(data):
    #     dt_c[i,:,:] = d[:,:]
    train_data = list(zip(train_sent_1,train_sent_2,train_len1,train_len2,label))
    batches = batch_iter(train_data,batch_size=60,epochs=epochs,Isshuffle=False)
    ## run the graph
    print("\n")
    i = 0
    max_acc = -1
    for batch in batches:
        sent_1,sent_2,len1,len2,y = zip(*batch)
        sent_1 = np.array(sent_1)
        sent_2 = np.array(sent_2)
        len1 = np.array(len1)
        len2 = np.array(len2)
        y = np.array(y)
        feed_dict = {
            m.sent1 :    sent_1,
            m.sent2     : sent_2,
            m.sent1_length : len1,
            m.sent2_length : len2,
            m.labels: y,
            m.dropout : 0.5
        }
        _,loss,accuracy = sess.run([train_op,m.loss,m.acc],feed_dict=feed_dict)
        print("step - "+str(i)+"    loss is " + str(loss)+" and accuracy is "+str(accuracy))
        sum_acc = 0
        sum_loss = 0

        if i%check_point == 0 and i > 0:
            j = 0
            test_batches = batch_iter(list(zip(test_data_1,test_data_2,test_length1,test_length2,test_labels)), batch_size=60, epochs=1,Isshuffle=False)
            for test_batch in test_batches:
                sent_1,sent_2,len1,len2,y = zip(*test_batch)
                sent_1 = np.array(sent_1)
                sent_2 = np.array(sent_2)

                len1 = np.array(len1)
                len2 = np.array(len2)
                y = np.array(y)
                feed_dict = {
                    m.sent1: sent_1,
                    m.sent2: sent_2,
                    m.sent1_length: len1,
                    m.sent2_length: len2,
                    m.labels: y,
                    m.dropout: 1.0
                }

                loss, accuracy = sess.run([m.loss, m.acc], feed_dict=feed_dict)
                sum_acc += accuracy
                sum_loss += loss
                j += 1
            print(" test loss is " + str(sum_loss / j) + " and test-accuracy is " + str(sum_acc / j))
            if sum_acc/j > max_acc:
                max_acc = sum_acc/j
                save_path = "saved_model/model-" + str(i)
                saver.save(sess, save_path=save_path)
                print("Model saved to " + save_path)


        i += 1
    print("maximum accuracy acheived is "+str(max_acc))
    return sess

word_vecs = word_embedings(debug=False)
batch_size = 60
embedding_size = 300


res_train = load_data(path=r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\sick2014\SICK_train.txt')

train_data_1 = res_train['data_1']
train_data_2 = res_train['data_2']
train_length1 = res_train['length1']
train_length2 = res_train['length2']
labels = res_train['label']
word2Id = res_train['word2Id']
Id2Word = res_train['Id2Word']
max_sequence_length = res_train['max_sequence_length']
vocab_size = res_train['vocab_size']
total_classes = res_train['total_classes']


res_test = load_test_data(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\sick2014\SICK_test_annotated.txt',max_sequence_length,word2Id,Id2Word)
word2Id = res_test['word2Id']
test_data_1 = res_test['data_1']
test_data_2 = res_test['data_2']
test_labels = res_test['label']
test_length1 = res_test['length1']
test_length2 = res_test['length2']
Id2Word = res_test['Id2Word']

Id2Vec = np.zeros([len(Id2Word.keys()),embedding_size])
words_list = word_vecs.word2vec.keys()
for i in range(len(Id2Word.keys())):
    word = Id2Word[i]
    if word in words_list:
        Id2Vec[i,:] = word_vecs.word2vec[word]
    else:
        Id2Vec[i, :] = word_vecs.word2vec['unknown']

m = model(max_sequence_length,total_classes,embedding_size,Id2Vec,batch_size)

train(m,train_data_1,train_data_2,train_length1,train_length2,labels,300,0.001,check_point=50)
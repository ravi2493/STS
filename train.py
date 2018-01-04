from simple_model import model
import numpy as np
import tensorflow as tf
from preprocess import load_data_chars
from glove import word_embedings
import random

def batch_iter(data, batch_size, epochs, Isshuffle=True):
    ## check inputs
    assert isinstance(batch_size,int)
    assert isinstance(epochs,int)
    assert isinstance(Isshuffle,bool)

    num_batches = int((len(data)-1)/batch_size) + 1
    ## data padded
    data = np.array(data+data[:2*batch_size])
    data_size = len(data)
    print("size of data"+str(data_size)+"---"+str(len(data)))
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

def train(m,data_1,data_chars_1,data_2,data_chars_2,label,epochs=800,learning_rate=0.001,check_point=200):
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
    sess.run(tf.global_variables_initializer())
    # dt = np.zeros([len(data),len(data[0])],dtype=int)
    # dt_c = np.zeros([len(data_chars),len(data_chars[0]),len(data_chars[0])])
    # for i,d in enumerate(data):
    #     dt[i,:] = d
    # for i,d in enumerate(data):
    #     dt_c[i,:,:] = d[:,:]
    train_data = list(zip(data_1,data_chars_1,data_2,data_chars_2,label))
    batches = batch_iter(train_data,batch_size=60,epochs=epochs,Isshuffle=True)
    ## run the graph
    print("\n")
    i = 0
    max_acc = -1
    for batch in batches:
        x_1,chars_x_1,x_2,chars_x_2,y = zip(*batch)
        x_1 = np.array(x_1)
        x_2 = np.array(x_2)
        chars_x_1 = np.array(chars_x_1)
        chars_x_2 = np.array(chars_x_2)
        y = np.array(y)
        feed_dict = {
            m.x1       : x_1,
            m.chars_x1 : chars_x_1,
            m.x2        : x_2,
            m.chars_x2  : chars_x_2,
            m.labels: y,
            m.dropout : 0.5
        }
        _,loss,accuracy = sess.run([train_op,m.loss,m.acc],feed_dict=feed_dict)
        print("step - "+str(i)+"    loss is " + str(loss)+" and accuracy is "+str(accuracy))
        sum_acc = 0
        sum_loss = 0

        if i%check_point == 0 and i > 0:
            j = 0
            test_batches = batch_iter(list(zip(test_data_1,test_data_chars_1,test_data_2,test_data_chars_2,test_labels)), batch_size=60, epochs=1)
            for test_batch in test_batches:
                x_1,chars_x_1,x_2,chars_x_2,y = zip(*test_batch)
                x_1 = np.array(x_1)
                chars_x_1 = np.array(chars_x_1)

                x_2 = np.array(x_2)
                chars_x_2 = np.array(chars_x_2)

                y = np.array(y)
                feed_dict = {
                    m.x1: x_1,
                    m.chars_x1: chars_x_1,
                    m.x2: x_2,
                    m.chars_x2: chars_x_2,
                    m.labels: y,
                    m.dropout : 1.0
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
    return sess

# this will load data from default path
word_vecs = word_embedings(debug=False)
batch_size = 60
embedding_size = 300

# train_paths = []
# test_paths = []

res = load_data_chars()
print("done")

data_1 = res['data_1']
data_chars_1 = res['data_char_1']
data_2 = res['data_2']
data_chars_2 = res['data_char_2']

label = res['label']
word2Id = res['word2Id']
char2Id = res['char2Id']
words_data_list = word2Id.keys()
Id2Word  = res['Id2Word']
max_sequence_length = res['max_sequence_length']
max_word_len = res['max_word_len']
total_classes = res['total_classes']

Id2Vec = np.zeros([len(Id2Word.keys()),embedding_size])
words_list = word_vecs.word2vec.keys()
for i in range(len(Id2Word.keys())):
    word = Id2Word[i]
    if word in words_list:
        Id2Vec[i,:] = word_vecs.word2vec[word]
    else:
        Id2Vec[i, :] = word_vecs.word2vec['unknown']





m = model(
          max_sequence_length=max_sequence_length,
          total_classes=1,
          embedding_size=300,
          char_size= len(char2Id.keys()),
          char_embed_size=9,
          id2Vecs= Id2Vec,
          batch_size=60,
          max_word_len=max_word_len,
          threshold=0.5
        )



## split data to train and test
n = len(data_1)
q = 0.05 # ratio of test and train
test_data_len = int(q*n)
a = random.sample(range(1,n),int(q*n)) ## ids for test data
test_data_1 = np.zeros([test_data_len,max_sequence_length])
test_data_chars_1 = np.zeros([test_data_len,max_sequence_length,max_word_len])
test_data_2 = np.zeros([test_data_len,max_sequence_length])
test_data_chars_2 = np.zeros([test_data_len,max_sequence_length,max_word_len])
test_labels = np.zeros([test_data_len,total_classes])
for i,e in enumerate(a):
    test_data_1[i,:] = data_1[e,:]
    test_data_chars_1[i,:,:] = data_chars_1[e,:,:]

    test_data_2[i,:] = data_2[e,:]
    test_data_chars_2[i,:,:] = data_chars_2[e,:,:]
    test_labels[i,:] = label[e,:]

train_data_len = n - test_data_len

train_data_1 = np.zeros([train_data_len,max_sequence_length])
train_data_chars_1 = np.zeros([train_data_len,max_sequence_length,max_word_len])

train_data_2 = np.zeros([train_data_len,max_sequence_length])
train_data_chars_2 = np.zeros([train_data_len,max_sequence_length,max_word_len])
train_labels = np.zeros([train_data_len,total_classes])
i = 0
for e in range(n):
    if e not in a:
        train_data_1[i,:] = data_1[e,:]
        train_data_chars_1[i, :, :] = data_chars_1[e, :, :]

        train_data_2[i,:] = data_2[e,:]
        train_data_chars_2[i, :, :] = data_chars_2[e, :, :]
        train_labels[i,:] = label[e,:]
        i += 1

train(m,data_1,data_chars_1,data_2,data_chars_2,train_labels,learning_rate=0.002)
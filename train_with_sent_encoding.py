from simple_model import model
import numpy as np
import tensorflow as tf
# from preprocess import load_data_chars,load_test_data_chars
# from glove import word_embedings
# import random
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

def train(m,train_sent_1,train_sent_2,label,epochs=200,learning_rate=0.001,check_point=100):

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
        log_device_placement=False
    )

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
    train_data = list(zip(train_sent_1,train_sent_2,label))
    batches = batch_iter(train_data,batch_size=60,epochs=epochs,Isshuffle=False)
    ## run the graph
    print("\n")
    i = 0
    max_acc = -1
    for batch in batches:
        sent_1,sent_2,y = zip(*batch)
        sent_1 = np.array(sent_1)
        sent_2 = np.array(sent_2)
        y = np.array(y)
        feed_dict = {
            m.sent1 :    sent_1,
            m.sent2 : sent_2,
            m.labels: y,
            m.dropout_keep_prob : 0.5
        }
        _,loss,accuracy = sess.run([train_op,m.loss,m.acc],feed_dict=feed_dict)
        print("step - "+str(i)+"    loss is " + str(loss)+" and accuracy is "+str(accuracy))
        sum_acc = 0
        sum_loss = 0

        if i%check_point == 0 and i > 0:
            j = 0

            feed_dict = {
            m.sent1 : train_sent_1,
            m.sent2 : train_sent_2,
            m.labels: label,
            m.dropout_keep_prob : 1.0
            }

            loss, accuracy = sess.run([m.loss, m.acc], feed_dict=feed_dict)
            print("train loss is " + str(loss) + " and accuracy is " + str(accuracy))

            test_batches = batch_iter(list(zip(test_sent_1,test_sent_2,test_labels)), batch_size=4926, epochs=1,Isshuffle=False)
            for test_batch in test_batches:
                sent_1,sent_2,y = zip(*test_batch)
                sent_1 = np.array(sent_1)
                sent_2 = np.array(sent_2)

                y = np.array(y)
                feed_dict = {
                    m.sent1: sent_1,
                    m.sent2: sent_2,
                    m.labels: y,
                    m.dropout_keep_prob : 1.0
                }

                loss, accuracy = sess.run([m.loss, m.acc], feed_dict=feed_dict)
                sum_acc += accuracy
                sum_loss += loss
                j += 1
            print(" test loss is " + str(sum_loss / j) + " and test-accuracy is " + str(sum_acc / j))
            if sum_acc/j > max_acc:
                max_acc = sum_acc/j
                # save_path = "saved_model/model-" + str(i)
                # saver.save(sess, save_path=save_path)
                # print("Model saved to " + save_path)
                print("better model")


        i += 1
    print("maximum accuracy acheived is "+str(max_acc))
    return sess



# train_paths = []
# train_paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\MSRpar.test.tsv')

# test_paths = []
# test_paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\MSRpar.train.tsv')

# res_train = load_data_chars(train_paths,score_position=0)
f1 = open(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data_pickles_from_nli\sts\sick2014\SICK_train.txt_1','rb')
# f2 = open(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data_pickles\sts\semeval-sts\2012\MSRpar.train.tsv','rb')
# f3 = open(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data_pickles\sts\semeval-sts\2012\MSRpar.test.tsv','rb')
train_sents_encoding_list_1 = pickle.load(f1)
# train_sents_encoding_list_2 = pickle.load(f2)
# train_sents_encoding_list_3 = pickle.load(f3)
train_data_len = len(train_sents_encoding_list_1[0]) \
                 # + len(train_sents_encoding_list_2[0]) + len(train_sents_encoding_list_3[0])
sent_dim = 8200
train_sent_1 = np.zeros([train_data_len,sent_dim])
train_sent_2 = np.zeros([train_data_len,sent_dim])

for i in range(len(train_sents_encoding_list_1[0])):
    train_sent_1[i,:] = train_sents_encoding_list_1[0][i,:,:]
    train_sent_2[i,:] = train_sents_encoding_list_1[1][i,:,:]

# i = len(train_sents_encoding_list_1[0])
# for j in range(len(train_sents_encoding_list_2[0])):
#     train_sent_1[i+j,:] = train_sents_encoding_list_2[0][j,:,:]
#     train_sent_2[i+j,:] = train_sents_encoding_list_2[1][j,:,:]
#
# i = i + len(train_sents_encoding_list_2[0])
# for j in range(len(train_sents_encoding_list_3[0])):
#     train_sent_1[i+j,:] = train_sents_encoding_list_3[0][j,:,:]
#     train_sent_2[i+j,:] = train_sents_encoding_list_3[1][j,:,:]

labels = train_sents_encoding_list_1[2]\
         # + train_sents_encoding_list_2[2] + train_sents_encoding_list_3[2]
# labels = np.reshape(labels,newshape=[len(labels),1])

f = open(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data_pickles_from_nli\sts\sick2014\SICK_test_annotated.txt_1','rb')

test_sents_encoding_list = pickle.load(f)
test_sent_1 = np.zeros([len(test_sents_encoding_list[0]),sent_dim])
test_sent_2 = np.zeros([len(test_sents_encoding_list[0]),sent_dim])
for i in range(len(test_sents_encoding_list[0])):
    test_sent_1[i,:] = test_sents_encoding_list[0][i,:,:]
    test_sent_2[i,:] = test_sents_encoding_list[1][i,:,:]
test_labels = test_sents_encoding_list[2]
# test_labels = np.reshape(test_labels,newshape=[len(test_labels),1])

m = model(
          max_sequence_length=10,
          total_classes=1,
          embedding_size=300,
          char_size= 100,
          char_embed_size=9,
          id2Vecs= np.zeros([10,100]),
          # batch_size=60,
          sent_dim=sent_dim,
          max_word_len=10,
          threshold=0.5
        )

train(m,train_sent_1,train_sent_2,labels,learning_rate=0.001)